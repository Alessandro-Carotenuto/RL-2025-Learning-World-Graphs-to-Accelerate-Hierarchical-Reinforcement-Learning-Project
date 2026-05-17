import random
import math
import time
import torch

# minigrid imports
from minigrid.core.world_object import Wall

# Local imports
from wrappers.minigrid_wrapper import MinigridWrapper, EnvModes, EnvSizes
from utils.statistics_buffer import StatBuffer
from local_networks.vaesystem import VAESystem
from local_networks.policy_networks import GoalConditionedPolicy
from utils.misc import manhattan_distance, resolve_device, _walk_away_from_spawn
from utils.checkpoint import save_phase1_checkpoint, load_phase1_checkpoint, restore_maze_from_grid_state
from utils.visualization import (plot_training_diagnostics, save_graph_visualization,
                                  create_phase1_gif, render_phase2_episode_gif,
                                  _run_and_save_episode, print_grid_image, _plot_phase1_run)
from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker, HierarchicalTrainer


def alternating_training_loop(env, policy, vae_system, buffer, max_iterations: int = 8, convergence_threshold: float = 0.01,
                              explore_top_fraction: float = 0.20,
                              diversity_walk_number: int = 10,
                              walk_length: int = 250,
                              walk_bias: float = 0.65,
                              walk_episodes: int = 4,
                              graph_walk_length: int = 20,
                              graph_num_attempts: int = 70,
                              spread_alpha: float = 0.0):
    """
    Main alternating training loop with persistent KL annealing.
    explore_top_fraction: fraction of pivotal states (sorted farthest-from-spawn first)
                          used for trajectory collection once COVERAGE_THRESHOLD is reached.
    diversity_walk_number: number of biased random walks per iteration for geographic diversity.
    walk_length: steps per diversity walk.
    walk_bias: probability of moving away from spawn at each walk step.
    walk_episodes: episodes collected from each walk destination.
    """
    print("Starting Alternating Training Loop:")
    print("=" * 50)

    reconstruction_losses = []
    all_pivotal_states = []
    pivotal_states = []
    metrics = {
        'num_pivotal_states_per_iteration': [],
        'policy_success_rates': [],
    }

    persistent_kl_weight = 1.0

    # Compute spawn position once (it's fixed throughout Phase 1)
    env.reset()
    spawn_pos = tuple(env.agent_pos)
    print(f"Spawn position: {spawn_pos}")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        print(f"Current KL weight: {persistent_kl_weight:.3f}")
        
        # Collect initial data if needed
        if buffer.episodes_in_buffer < 3:
            print("Not enough episodes for VAE training, collecting initial data...")
            for _ in range(5):
                env.reset()
                start_pos = tuple(env.agent_pos)
                episodes = policy.collect_episodes_from_position(
                    env, start_pos, num_episodes=6, max_episode_length=100, vae_system=vae_system
                )
                if episodes:
                    buffer.add_episodes(episodes)
            continue
        
        # Train VAE with persistent KL weight
        print(f"First Half: Training VAE on {buffer.episodes_in_buffer} episodes...")
        try:
            pivotal_states = vae_system.train_vae(
                buffer,
                num_epochs=25,
                batch_size=8,
                initial_kl_weight=persistent_kl_weight,
                annealing_rate=0.0,  # No annealing within iteration
                spread_alpha=spread_alpha
            )
        except Exception as e:
            print(f"VAE training failed: {e}")
            continue
        
        # Track metrics
        if vae_system.training_history:
            current_loss = vae_system.training_history[-1]['reconstruction_loss']
            reconstruction_losses.append(current_loss)
            print(f"Current reconstruction loss: {current_loss:.4f}")
        
        metrics['num_pivotal_states_per_iteration'].append(len(pivotal_states))

        all_pivotal_states.append(pivotal_states.copy())
        print(f"Discovered {len(pivotal_states)} pivotal states: {pivotal_states[:3]}...")
        
        # Phase 2: Collect trajectories from pivotal states
        # Sort by distance from spawn descending — farthest states first — to break the
        # self-reinforcing clustering loop that keeps all pivotal states near spawn.
        COVERAGE_THRESHOLD = 50
        sorted_by_dist = sorted(pivotal_states, key=lambda s: manhattan_distance(s, spawn_pos), reverse=True)
        if len(pivotal_states) < COVERAGE_THRESHOLD:
            states_to_explore = sorted_by_dist
            print(f"Second Half: Collecting from ALL {len(pivotal_states)} pivotal states sorted farthest-first from spawn {spawn_pos}...")
        else:
            top_n = max(1, int(len(pivotal_states) * explore_top_fraction))
            states_to_explore = sorted_by_dist[:top_n]
            pct = int(explore_top_fraction * 100)
            print(f"Second Half: top {pct}% farthest from spawn ({top_n}/{len(pivotal_states)}) pivotal states...")

        episodes_collected = 0
        success_count = 0
        for i, start_state in enumerate(states_to_explore):
            print(f"  Collecting from pivotal state {i+1}/{len(states_to_explore)}: {start_state}")
            
            curiosity_weight = 0.0 if iteration == 0 else max(0.15, 0.5 - (iteration * 0.05))
            
            try:
                episodes = policy.collect_episodes_from_position(
                    env, start_state,
                    num_episodes=10,
                    max_episode_length=100,
                    vae_system=vae_system,
                    curiosity_weight=curiosity_weight
                )
                
                if episodes:
                    buffer.add_episodes(episodes)
                    episodes_collected += len(episodes)
                    success_count += sum(1 for ep in episodes if ep.get('goal_reached', False))
            except Exception as e:
                print(f"  Failed to collect from {start_state}: {e}")
                continue
            
        # Track success rate
        if episodes_collected > 0:
            success_rate = success_count / episodes_collected
            metrics['policy_success_rates'].append(success_rate)
        else:
            metrics['policy_success_rates'].append(0.0)


        print(f"  Collected {episodes_collected} new episodes")
        print(f"  Total episodes in buffer: {buffer.episodes_in_buffer}")
        
        # Diversity collection: biased random walks away from spawn to break clustering.
        # Each walk physically navigates to a distant region (no map knowledge used).
        _cw = 0.0 if iteration == 0 else max(0.15, 0.5 - (iteration * 0.05))
        print(f"Spatial diversity: {diversity_walk_number} biased walks from spawn {spawn_pos}...")
        for _wi in range(diversity_walk_number):
            _dst = _walk_away_from_spawn(env, spawn_pos, walk_length=walk_length, bias=walk_bias)
            dist_from_spawn = manhattan_distance(_dst, spawn_pos)
            print(f"  Walk {_wi+1}: reached {_dst} (dist={dist_from_spawn})")
            try:
                _eps = policy.collect_episodes_from_position(
                    env, _dst, num_episodes=walk_episodes, max_episode_length=100,
                    vae_system=vae_system, curiosity_weight=_cw
                )
                if _eps:
                    buffer.add_episodes(_eps)
            except Exception as e:
                print(f"  Walk {_wi+1} collection failed: {e}")

        # Check for convergence (not before min_iterations to ensure coverage)
        MIN_ITERATIONS = 5
        if len(reconstruction_losses) >= 3 and iteration + 1 >= MIN_ITERATIONS:
            recent_losses = reconstruction_losses[-3:]
            loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
            avg_change = sum(loss_changes) / len(loss_changes)

            print(f"Average loss change over last 3 iterations: {avg_change:.5f}")

            if avg_change < convergence_threshold:
                print("Reconstruction loss has plateaued - training converged!")
                break
    

    # Construct world graph
    world_graph = policy.complete_world_graph_discovery(env, pivotal_states,
                                                         graph_walk_length=graph_walk_length,
                                                         graph_num_attempts=graph_num_attempts)
    
    # Final summary
    print(f"\nAlternating Training Complete!")
    print(f"Total iterations: {len(reconstruction_losses)}")
    print(f"Final episodes in buffer: {buffer.episodes_in_buffer}")
    print(f"Final pivotal states ({len(pivotal_states)}): {pivotal_states}")
    
    return pivotal_states, world_graph, metrics, all_pivotal_states

# DIAGNOSTIC FUNCTIONS ----------------------------------------------------

def diagnose_graph_connectivity(world_graph, pivotal_states, env):
    """
    Diagnose graph connectivity issues from spawn position.
    """
    print("\n" + "="*70)
    print("GRAPH CONNECTIVITY DIAGNOSTICS")
    print("="*70)
    
    # Get spawn position
    env.phase = 2
    env.reset()
    spawn_pos = tuple(env.agent_pos)
    
    print(f"\nAgent spawn position: {spawn_pos}")
    print(f"Is spawn a pivotal state? {spawn_pos in pivotal_states}")
    
    # Check if spawn is in graph
    print(f"Is spawn in graph nodes? {spawn_pos in world_graph.nodes}")
    
    # Check connectivity from spawn to all pivotal states
    print(f"\nChecking paths from spawn to all {len(pivotal_states)} pivotal states:")
    reachable_from_spawn = []
    unreachable_from_spawn = []
    
    for pivotal in pivotal_states:
        if pivotal == spawn_pos:
            print(f"  {pivotal}: SPAWN (skip)")
            continue
            
        path, distance = world_graph.shortest_path(spawn_pos, pivotal)
        
        if path and distance < float('inf'):
            reachable_from_spawn.append((pivotal, distance))
            if len(reachable_from_spawn) <= 5:  # Show first 5
                print(f"  {pivotal}: REACHABLE (distance={distance:.0f})")
        else:
            unreachable_from_spawn.append(pivotal)
            if len(unreachable_from_spawn) <= 5:  # Show first 5
                print(f"  {pivotal}: UNREACHABLE (no path in graph)")
    
    print(f"\nSummary:")
    print(f"  Reachable pivotal states: {len(reachable_from_spawn)}/{len(pivotal_states)-1}")
    print(f"  Unreachable pivotal states: {len(unreachable_from_spawn)}/{len(pivotal_states)-1}")
    
    if len(unreachable_from_spawn) > 0:
        print(f"\n⚠ WARNING: {len(unreachable_from_spawn)} pivotal states unreachable from spawn!")
        print(f"  Manager may select goals Worker cannot traverse to.")
    
    # Check if graph is generally connected
    print(f"\nChecking overall graph connectivity:")
    total_pairs = len(pivotal_states) * (len(pivotal_states) - 1)
    connected_pairs = 0
    
    for i, start in enumerate(pivotal_states):
        for j, end in enumerate(pivotal_states):
            if i != j:
                path, _ = world_graph.shortest_path(start, end)
                if path:
                    connected_pairs += 1
    
    if total_pairs > 0:
        connectivity_pct = 100 * connected_pairs / total_pairs
        print(f"  Connected pairs: {connected_pairs}/{total_pairs} ({connectivity_pct:.1f}%)")
    else:
        connectivity_pct = 100.0
        print(f"  Only one pivotal state, trivially connected.")
    
    if connectivity_pct < 50:
        print(f"  ⚠ WARNING: Graph is poorly connected!")
    
    print("="*70 + "\n")
    
    return reachable_from_spawn, unreachable_from_spawn

def diagnose_worker_behavior_single_episode(env, manager, worker, world_graph):
    """
    Run ONE episode with detailed Worker diagnostics.
    """
    print("\n" + "="*70)
    print("WORKER BEHAVIOR DIAGNOSTICS - SINGLE EPISODE")
    print("="*70)
    
    env.phase = 2
    obs = env.reset()
    start_pos = tuple(env.agent_pos)
    
    print(f"\nAgent starts at: {start_pos}")
    print(f"Balls at: {list(env.active_balls)[:5]}")
    
    manager.reset_manager_state()
    worker.reset_worker_state()
    
    episode_diagnostics = []
    
    for horizon_num in range(5):  # Just 5 horizons for diagnosis
        print(f"\n--- Horizon {horizon_num} ---")
        
        current_pos = tuple(env.agent_pos)
        print(f"Start position: {current_pos}")
        
        # Manager selects goals
        wide_goal, narrow_goal, _, _, _ = manager.get_manager_action(current_pos)
        print(f"Manager goals: wide={wide_goal}, narrow={narrow_goal}")
        
        # Check if wide goal is reachable
        dist_to_wide = manhattan_distance(current_pos, wide_goal)
        print(f"  Distance to wide goal: {dist_to_wide}")
        
        # Check if Worker should traverse
        should_traverse = worker.should_traverse(current_pos, wide_goal)
        print(f"  Worker should_traverse: {should_traverse}")
        
        if should_traverse:
            path = worker.plan_traversal(current_pos, wide_goal)
            print(f"  Planned traversal path: {path}")
        else:
            # Why not?
            is_at_pivotal = worker.is_at_pivotal_state(current_pos)
            is_wide_pivotal = worker.is_at_pivotal_state(wide_goal)
            graph_path, graph_dist = world_graph.shortest_path(current_pos, wide_goal)
            
            print(f"  Why no traversal?")
            print(f"    Current is pivotal: {is_at_pivotal}")
            print(f"    Wide goal is pivotal: {is_wide_pivotal}")
            print(f"    Graph path exists: {graph_path is not None}")
        
        # Execute Worker for horizon
        positions_visited = [current_pos]
        actions_taken = []
        
        for h in range(10):
            action, log_prob, value = worker.get_action(current_pos, wide_goal, narrow_goal,agent_dir=env.agent_dir)
            actions_taken.append(action)
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                current_pos = tuple(env.agent_pos)
                positions_visited.append(current_pos)
            except Exception:
                break
            
            # Check if reached goals
            if current_pos == wide_goal:
                print(f"  ✓ Reached wide goal at step {h+1}")
                break
            if current_pos == narrow_goal:
                print(f"  ✓ Reached narrow goal at step {h+1}")
                break
            
            if terminated or truncated:
                break
        
        # Analyze Worker trajectory
        final_dist_to_wide = manhattan_distance(current_pos, wide_goal)
        final_dist_to_narrow = manhattan_distance(current_pos, narrow_goal)
        
        print(f"  Worker trajectory: {len(positions_visited)} positions")
        print(f"  Final distance to wide: {final_dist_to_wide} (started at {dist_to_wide})")
        print(f"  Final distance to narrow: {final_dist_to_narrow}")
        print(f"  Actions: {actions_taken}")
        
        horizon_diagnostics = {
            'start_pos': start_pos,
            'wide_goal': wide_goal,
            'narrow_goal': narrow_goal,
            'initial_distance': dist_to_wide,
            'final_distance': final_dist_to_wide,
            'should_traverse': should_traverse,
            'positions_visited': positions_visited,
            'actions_taken': actions_taken
        }
        episode_diagnostics.append(horizon_diagnostics)
    
    print("\n" + "="*70)
    return episode_diagnostics

def test_phase1_with_diagnostics(config=None):
    """
    Test Phase 1 using alternating_training_loop with diagnostic tracking. 
    """
    default_config = {
        'maze_size': EnvSizes.MEDIUM,
        'phase1_iterations': 15,
        'vae_mu0': 10.0,
        'goal_policy_lr': 5e-3,
        'device': resolve_device()
    }

    if config is not None:
        default_config.update(config)
    config = default_config
    
    print("PHASE 1 DIAGNOSTIC TEST")
    print("="*70)
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Setup
    env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL,phase_one_eps=config['phase1_iterations']*1000)
    env.phase = 1
    env.randomgen = True
    
    policy = GoalConditionedPolicy(lr=config['goal_policy_lr'], device=config['device'])
    vae_system = VAESystem(
        state_dim=16, 
        action_vocab_size=7, 
        mu0=config['vae_mu0'], 
        grid_size=env.size,
        device=config['device']
    )
    buffer = StatBuffer()
    
    # Run alternating training (now with persistent KL)
    pivotal_states, world_graph, loop_metrics, all_pivotal_states_history = alternating_training_loop(
        env, policy, vae_system, buffer,
        max_iterations=config['phase1_iterations'],
        explore_top_fraction=config.get('explore_top_fraction', 0.20),
        diversity_walk_number=config.get('diversity_walk_number', 10),
        walk_length=config.get('walk_length', 250),
        walk_bias=config.get('walk_bias', 0.65),
        walk_episodes=config.get('walk_episodes', 4),
        graph_walk_length=config.get('graph_walk_length', 20),
        graph_num_attempts=config.get('graph_num_attempts', 70),
        spread_alpha=config.get('pivotal_spread_alpha', 0.0)
    )
    
    # Extract metrics from VAE training history
    metrics = {
        'vae_losses': [h['total_loss'] for h in vae_system.training_history],
        'vae_reconstruction': [h['reconstruction_loss'] for h in vae_system.training_history],
        'vae_kl': [h['kl_divergence'] for h in vae_system.training_history],
        'vae_l0': [h['expected_l0'] for h in vae_system.training_history],
        'num_pivotal_states': loop_metrics['num_pivotal_states_per_iteration'],
        'policy_episodes': buffer.episodes_in_buffer,
        'policy_success_rate': loop_metrics['policy_success_rates']
    }
    
    # Graph statistics
    graph_stats = None
    if len(pivotal_states) > 0:
        graph_stats = {
            'nodes': len(world_graph.nodes),
            'edges': len(world_graph.edges),
            'connectivity': 0
        }
        
        # Check connectivity
        connected_pairs = 0
        total_pairs = len(pivotal_states) * (len(pivotal_states) - 1)
        for i, start in enumerate(pivotal_states):
            for j, end in enumerate(pivotal_states):
                if i != j:
                    path, dist = world_graph.shortest_path(start, end)
                    if path:
                        connected_pairs += 1
        
        graph_stats['connectivity'] = connected_pairs / total_pairs if total_pairs > 0 else 0
        
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {graph_stats['nodes']}")
        print(f"  Edges: {graph_stats['edges']}")
        print(f"  Connectivity: {graph_stats['connectivity']*100:.1f}%")


    GRIDSTATE = env.getGridState()

    _plot_phase1_run(vae_system, loop_metrics, config['vae_mu0'],
                     f"mu0={config['vae_mu0']:.1f}",
                     f'phase1_diagnostics_mu{config["vae_mu0"]:.1f}.png')

    save_graph_visualization(world_graph, pivotal_states, config['vae_mu0'], grid_state=GRIDSTATE)
    create_phase1_gif(all_pivotal_states_history, GRIDSTATE)

    checkpoint_path = f"phase1_checkpoint_{config['maze_size'].name}.pt"
    save_phase1_checkpoint(checkpoint_path, pivotal_states, world_graph, policy, vae_system, config, GRIDSTATE)

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 1 TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Episodes collected: {metrics['policy_episodes']}")
    print(f"Final pivotal states: {len(pivotal_states)}")
    if graph_stats:
        print(f"Graph connectivity: {graph_stats['connectivity']*100:.1f}%")
    if metrics['vae_losses']:
        print(f"Final VAE loss: {metrics['vae_losses'][-1]:.4f}")
        print(f"Final KL divergence: {metrics['vae_kl'][-1]:.4f}")

    
    return {
        'metrics': metrics,
        'pivotal_states': pivotal_states,
        'world_graph': world_graph,
        'graph_stats': graph_stats,
        'buffer': buffer,
        'policy': policy,
        'vae_system': vae_system
    }

def analyze_phase1_metrics(results):
    """Extract key metrics and diagnose Phase 1 issues.
    Expects results from test_phase1_with_diagnostics() — not from run_phase1_comparison()."""
    
    metrics = results['metrics']
    graph_stats = results['graph_stats']
    
    # Extract final values
    final_vae_loss = metrics['vae_losses'][-1] if metrics['vae_losses'] else None
    final_recon = metrics['vae_reconstruction'][-1] if metrics['vae_reconstruction'] else None
    final_l0 = metrics['vae_l0'][-1] if metrics['vae_l0'] else None
    final_success = metrics['policy_success_rate'][-1] if metrics['policy_success_rate'] else None
    
    # Convergence checks
    vae_converged = False
    if len(metrics['vae_losses']) >= 3:
        recent = metrics['vae_losses'][-3:]
        vae_converged = (max(recent) - min(recent)) < 0.01
    
    l0_on_target = False
    if final_l0 and results.get('vae_system'):
        target = results['vae_system'].mu0
        l0_on_target = abs(final_l0 - target) < target * 0.2  # Within 20%
    
    # Issue detection
    issues = []
    warnings = []
    
    if final_vae_loss and final_vae_loss > 10:
        issues.append(f"VAE loss very high ({final_vae_loss:.2f}) - may not converge")
    
    if final_recon and final_recon > 5:
        issues.append(f"Reconstruction loss high ({final_recon:.2f}) - poor action prediction")
    
    if not l0_on_target and final_l0:
        target = results['vae_system'].mu0
        warnings.append(f"L0 ({final_l0:.1f}) far from target ({target:.1f}) - adjust mu0 or training")
    
    if graph_stats and graph_stats['connectivity'] < 0.3:
        issues.append(f"Graph poorly connected ({graph_stats['connectivity']*100:.1f}%) - pivotal states isolated")
    
    if final_success and final_success < 0.3:
        warnings.append(f"Low policy success ({final_success*100:.1f}%) - goals may be too far")
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 1 METRICS ANALYSIS")
    print("="*70)
    
    print("\n[VAE Performance]")
    print(f"  Final loss: {final_vae_loss:.4f}" if final_vae_loss else "  No data")
    print(f"  Reconstruction: {final_recon:.4f}" if final_recon else "  No data")
    print(f"  L0 sparsity: {final_l0:.2f}" if final_l0 else "  No data")
    print(f"  Converged: {'✓' if vae_converged else '✗'}")
    print(f"  L0 on target: {'✓' if l0_on_target else '✗'}")
    
    if graph_stats:
        print("\n[Graph Quality]")
        print(f"  Nodes: {graph_stats['nodes']}")
        print(f"  Edges: {graph_stats['edges']}")
        print(f"  Connectivity: {graph_stats['connectivity']*100:.1f}%")
        print(f"  Avg edges/node: {graph_stats['edges']/graph_stats['nodes']:.1f}" if graph_stats['nodes'] > 0 else "  N/A")
    
    print("\n[Policy Performance]")
    print(f"  Final success rate: {final_success*100:.1f}%" if final_success else "  No data")
    print(f"  Total episodes: {metrics['policy_episodes']}")
    
    if issues:
        print("\n[❌ ISSUES]")
        for issue in issues:
            print(f"  • {issue}")
    
    if warnings:
        print("\n[⚠️  WARNINGS]")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not issues and not warnings:
        print("\n[✓ All checks passed]")
    
    return {
        'final_vae_loss': final_vae_loss,
        'final_reconstruction': final_recon,
        'final_l0': final_l0,
        'vae_converged': vae_converged,
        'l0_on_target': l0_on_target,
        'graph_stats': graph_stats,
        'issues': issues,
        'warnings': warnings
    }

def compare_phase1_runs(runs_dict):
    """Print a comparison table across multiple Phase 1 runs.
    Expects {name: results} where each results comes from test_phase1_with_diagnostics().
    Do NOT pass the output of run_phase1_comparison() — it uses a different metrics structure."""
    
    print("\n" + "="*70)
    print("PHASE 1 COMPARISON")
    print("="*70)
    
    # Header
    print(f"\n{'Config':<20} {'Loss':<10} {'Recon':<10} {'L0':<8} {'Nodes':<8} {'Conn%':<8} {'Succ%':<8}")
    print("-" * 70)
    
    # Rows
    for name, results in runs_dict.items():
        m = results['metrics']
        g = results['graph_stats']
        
        loss = m['vae_losses'][-1] if m['vae_losses'] else float('nan')
        recon = m['vae_reconstruction'][-1] if m['vae_reconstruction'] else float('nan')
        l0 = m['vae_l0'][-1] if m['vae_l0'] else float('nan')
        nodes = g['nodes'] if g else 0
        conn = g['connectivity']*100 if g else 0
        succ = m['policy_success_rate'][-1]*100 if m['policy_success_rate'] else 0
        
        print(f"{name:<20} {loss:<10.3f} {recon:<10.3f} {l0:<8.1f} {nodes:<8} {conn:<8.1f} {succ:<8.1f}")

def _run_phase2_training(config, pivotal_states, world_graph, policy, env,
                         agent_start, first_balls, session_path, grid_state,
                         phase2_animation=True):
    
    manager = HierarchicalManager(
        pivotal_states,
        neighborhood_size=config['neighborhood_size'],
        lr=config['manager_lr'],
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],
        diagnostic_checkstart=config['diagnostic_checkstart'],
        device=config['device'],
    )
    worker = HierarchicalWorker(
        world_graph,
        pivotal_states,
        lr=config['worker_lr'],
        goal_policy=policy,
        device=config['device'],
    )
    manager.initialize_from_goal_policy(policy)
    worker.initialize_from_goal_policy(policy)

    print("\nDiagnosing Worker behavior BEFORE training:")
    diagnose_worker_behavior_single_episode(env, manager, worker, world_graph)

    env.phase = 2
    trainer = HierarchicalTrainer(
        manager, worker, env,
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],
        diagnostic_checkstart=config['diagnostic_checkstart'],
        goal_timeout=config.get('goal_timeout', 3),
    )

    print("\nPHASE 2: Hierarchical Training")
    metrics = {
        'rewards': [], 'steps': [], 'manager_updates': [],
        'worker_updates': [], 'times': [],
    }

    debug_interval = max(1, config['phase2_episodes'] // 20)
    for episode in range(config['phase2_episodes']):
        ep_start = time.time()
        stats = trainer.train_episode(
            max_steps=config['max_steps_per_episode'],
            full_breakdown_every=debug_interval,
        )
        metrics['rewards'].append(stats['episode_reward'])
        metrics['steps'].append(stats['episode_steps'])
        metrics['manager_updates'].append(stats['manager_updates'])
        metrics['worker_updates'].append(stats['worker_updates'])
        metrics['times'].append(time.time() - ep_start)
        if episode % debug_interval == 0 and episode > 0:
            print(f"\n--- Episode {episode+1}/{config['phase2_episodes']} | reward={stats['episode_reward']:.2f} | entropy={stats['manager_entropy']:.3f} | balls={stats['balls_collected']}/{trainer.env.total_balls} ---")

    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    plot_training_diagnostics(trainer, config)

    torch.save({
        'agent_start': agent_start,
        'ball_positions': first_balls,
        'manager_state_dict': manager.state_dict(),
        'worker_state_dict': worker.state_dict(),
        'goal_policy_state_dict': policy.state_dict(),
    }, session_path)
    print(f"Session updated with trained weights: '{session_path}'")

    if phase2_animation:
        _run_and_save_episode(
            manager, worker, config, grid_state, agent_start, first_balls,
            'phase2_final_episode.mp4', fps=15, max_steps=500,
            world_graph=world_graph, pivotal_states=pivotal_states,
        )

    return metrics

def run_phase2_standalone(
        checkpoint_path='phase1_checkpoint.pt',
        config_overrides=None,
        fixed_balls=True,
        phase2_animation=True):
    
    """Run Phase 2 training using a saved Phase 1 checkpoint.
    fixed_balls=True : same ball positions every episode (manager can learn spatial strategy)
    fixed_balls=False: random ball positions every episode
    """
    pivotal_states, world_graph, policy, vae_system, config, grid_state = load_phase1_checkpoint(checkpoint_path)

    if config_overrides:
        config.update(config_overrides)
        resolve_device(config)
        vae_system.to(config['device'])
        policy.to(config['device'])

    env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL, max_steps=config['max_steps_per_episode'])
    env.reset()
    restore_maze_from_grid_state(env, grid_state)

    # Find a valid start position: not a wall, ≥6 reachable cells
    attempt = 0
    while True:
        attempt += 1
        x = random.randint(1, env.size - 2)
        y = random.randint(1, env.size - 2)
        if isinstance(env.grid.get(x, y), Wall):
            continue
        reachables = env.BFS_all_reachable((x, y))
        if len(reachables) >= 6:
            env.agent_start_pos = (x, y)
            env.agent_pos = (x, y)
            env.placeable_grid[x][y] = False
            break

    agent_start = (x, y)
    print(f"Valid start found after {attempt} attempt(s): {agent_start} ({len(reachables)} reachable cells)")

    env.firstgen = False

    reachable = world_graph.get_reachable_nodes(agent_start)
    print(f"Graph reachability from {agent_start}: {len(reachable)}/{len(world_graph.nodes)} nodes reachable")
    unreachable = [n for n in pivotal_states if n not in reachable]
    print(f"  Unreachable from start: {unreachable[:10]}{'...' if len(unreachable) > 10 else ''}")

    if fixed_balls:
        first_balls = env.ResetMultiGoals(agent_start, goals=config.get('num_balls', 5))
        env.fixed_ball_positions = first_balls
        print(f"Fixed ball positions: {first_balls}")
    else:
        first_balls = None
        print("Ball positions: random each episode")

    session_path = checkpoint_path.replace('.pt', '_session.pt')
    metrics = _run_phase2_training(
        config, pivotal_states, world_graph, policy, env,
        agent_start, first_balls, session_path, grid_state, phase2_animation,
    )

    print(f"Best reward: {max(metrics['rewards']):.2f}")
    print(f"Final 10-ep avg: {sum(metrics['rewards'][-10:]) / 10:.2f}")
    return metrics


# ACTUAL TRAINING CODE ----------------------------------------------------
steps=2000

externalconfig = {
        'maze_size': EnvSizes.MEDIUM,
        'phase1_iterations': 3,
        'phase2_episodes': 200,
        'num_balls': 5,
        'max_steps_per_episode': steps,
        'manager_horizon': steps//250,
        'neighborhood_size': math.ceil(24/8),
        'manager_lr': 5e-4,
        'worker_lr': 1e-4,
        'goal_policy_lr': 5e-3,
        'vae_mu0': 9.0,
        'diagnostic_interval': 10000,
        'diagnostic_checkstart': False,
        'full_breakdown_every': 10,
        'goal_timeout': 15,             # max horizons before forcing a new Manager goal (horizon*timeout = max steps per goal)
        'pivotal_spread_alpha': 0.02,   # Phase 1: spread incentive for pivotal state selection (0=off, ~0.05=strong)
        'explore_top_fraction': 0.20,   # Phase 1: top % of pivotal states (by dist from spawn) used for trajectory collection
        'diversity_walk_number': 30,    # Phase 1: biased random walks per iteration
        'walk_length': 400,             # Phase 1: steps per diversity walk
        'walk_bias': 0.70,              # Phase 1: probability of stepping away from spawn
        'walk_episodes': 10,             # Phase 1: episodes collected per walk destination
        'graph_walk_length': 50,         # Phase 1: max steps per random walk for edge discovery
        'graph_num_attempts': 150,       # Phase 1: random walk attempts per pivotal state
        'convergence_threshold': 0.01,   # Phase 1: early-stop when avg loss change drops below this
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def train_full_phase1_phase2(
        config=externalconfig,
        phase1_animation=True,
        phase2_animation=True):
    
    """Complete training with comprehensive diagnostics."""
    # Hyperparameters setted up in externalconfig

    resolve_device(config)
    
    print("="*70)
    print("FULL TRAINING with Diagnostics")
    print("="*70)
    
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Phase 1
    env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL, max_steps=config['max_steps_per_episode'],phase_one_eps=config['phase1_iterations']*10000)
    env.phase = 1
    env.randomgen = True
    
    policy = GoalConditionedPolicy(lr=config['goal_policy_lr'], device=config['device'])
    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=config['vae_mu0'], grid_size=env.size)
    buffer = StatBuffer()
    
    print("\nPHASE 1: World Graph Discovery")
    start_time = time.time()
    
    pivotal_states, world_graph, stat_buffer, all_pivotal_states = alternating_training_loop(
        env, policy, vae_system, buffer, max_iterations=config['phase1_iterations'],
        convergence_threshold=config.get('convergence_threshold', 0.01),
        explore_top_fraction=config.get('explore_top_fraction', 0.20),
        diversity_walk_number=config.get('diversity_walk_number', 10),
        walk_length=config.get('walk_length', 250),
        walk_bias=config.get('walk_bias', 0.65),
        walk_episodes=config.get('walk_episodes', 4),
        graph_walk_length=config.get('graph_walk_length', 20),
        graph_num_attempts=config.get('graph_num_attempts', 70),
        spread_alpha=config.get('pivotal_spread_alpha', 0.0)
    )
    
    phase1_time = time.time() - start_time
    print(f"\nPhase 1 complete in {phase1_time:.1f}s")
    print(f"  Pivotal states: {len(pivotal_states)}")
    print(f"  Graph edges: {len(world_graph.edges)}")
    


    # Diagnose graph connectivity
    reachable, unreachable = diagnose_graph_connectivity(
        world_graph, pivotal_states, env
    )

    GRIDSTATE=env.getGridState()
    save_graph_visualization(world_graph, pivotal_states, config['vae_mu0'], grid_state=GRIDSTATE)

    if phase1_animation:
        create_phase1_gif(all_pivotal_states, GRIDSTATE)

    if len(pivotal_states) < 2:
        print(f"\nERROR: Phase 1 produced only {len(pivotal_states)} pivotal state(s). "
              f"Phase 2 requires at least 2. Check VAE training — try increasing phase1_iterations or vae_mu0.")
        return

    checkpoint_path = f"phase1_checkpoint_{config['maze_size'].name}.pt"
    save_phase1_checkpoint(checkpoint_path, pivotal_states, world_graph, policy, vae_system, config, GRIDSTATE)

    session_path = checkpoint_path.replace('.pt', '_session.pt')
    metrics = _run_phase2_training(
        config, pivotal_states, world_graph, policy, env,
        env.agent_start_pos, None, session_path, GRIDSTATE, phase2_animation,
    )

    print(f"Phase 1 time: {phase1_time:.1f}s")
    print(f"Phase 2 time: {sum(metrics['times']):.1f}s")
    print(f"Best reward: {max(metrics['rewards']):.2f}")
    print(f"Final 10-ep avg: {sum(metrics['rewards'][-10:])/10:.2f}")
    print(f"Avg manager updates/ep: {sum(metrics['manager_updates'])/len(metrics['manager_updates']):.1f}")
    print(f"Avg worker updates/ep: {sum(metrics['worker_updates'])/len(metrics['worker_updates']):.1f}")

def run_phase1_comparison(mu0_values=None, maze_size=EnvSizes.MEDIUM, iterations=50):
    """Run Phase 1 training on the same map with different mu0 values."""
    if mu0_values is None:
        mu0_values = [3.0, 6.0, 9.0]

    device = resolve_device()

    base_env = MinigridWrapper(size=maze_size, mode=EnvModes.MULTIGOAL, phase_one_eps=iterations * 10000)
    base_env.phase = 1
    base_env.randomgen = True
    base_env.reset()
    print_grid_image(base_env.getGridState(), name='base_map')
    base_env.randomgen = False
    base_env.firstgen = False

    results = {}

    for mu0 in mu0_values:
        print(f"\n{'='*70}\nRunning Phase 1 with mu0={mu0}\n{'='*70}")

        policy = GoalConditionedPolicy(lr=externalconfig['goal_policy_lr'], device=device)
        vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=mu0, grid_size=base_env.size, device=device)
        buffer = StatBuffer()
        base_env.phase = 1

        pivotal_states, world_graph, metrics, _ = alternating_training_loop(
            base_env, policy, vae_system, buffer, max_iterations=iterations
        )

        final_loss = vae_system.training_history[-1]['total_loss']
        results[mu0] = {
            'pivotal_states': pivotal_states,
            'world_graph': world_graph,
            'metrics': metrics,
            'final_loss': final_loss,
        }

        _plot_phase1_run(vae_system, metrics, mu0, f'mu0={mu0}', f'phase1_comparison_mu{mu0:.1f}.png')
        save_graph_visualization(world_graph, pivotal_states, mu0)
        print(f"Completed mu0={mu0} | pivotal states: {len(pivotal_states)} | edges: {len(world_graph.edges)}")

    print(f"\n{'='*70}\nCOMPARISON SUMMARY\n{'='*70}")
    print(f"{'mu0':<8} {'Pivotal':<10} {'Edges':<8} {'Final Loss':<12}")
    print("-" * 70)
    for mu0, res in results.items():
        print(f"{mu0:<8.1f} {len(res['pivotal_states']):<10} {len(res['world_graph'].edges):<8} {res['final_loss']:<12.4f}")

    return results

def run_phase1_size_comparison(sizes=None, mu0=9.0, iterations=50):
    """Run Phase 1 training on different environment sizes."""
    if sizes is None:
        sizes = [EnvSizes.SMALL, EnvSizes.MEDIUM]

    device = resolve_device()

    results = {}

    for size in sizes:
        print(f"\n{'='*70}\nRunning Phase 1 with size={size.name} (grid={size.value}x{size.value})\n{'='*70}")

        env = MinigridWrapper(size=size, mode=EnvModes.MULTIGOAL, phase_one_eps=iterations * 10000)
        env.phase = 1
        env.randomgen = True
        env.reset()
        print_grid_image(env.getGridState(), name=f'map_{size.name}')

        policy = GoalConditionedPolicy(lr=externalconfig['goal_policy_lr'], device=device)
        vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=mu0, grid_size=env.size, device=device)
        buffer = StatBuffer()

        pivotal_states, world_graph, metrics, _ = alternating_training_loop(
            env, policy, vae_system, buffer, max_iterations=iterations
        )

        final_loss = vae_system.training_history[-1]['total_loss']
        results[size.name] = {
            'pivotal_states': pivotal_states,
            'world_graph': world_graph,
            'metrics': metrics,
            'final_loss': final_loss,
            'grid_size': size.value,
        }

        _plot_phase1_run(vae_system, metrics, mu0, size.name, f'phase1_size_{size.name}.png')
        save_graph_visualization(world_graph, pivotal_states, mu0)
        print(f"Completed {size.name} | pivotal states: {len(pivotal_states)} | edges: {len(world_graph.edges)}")

    print(f"\n{'='*70}\nSIZE COMPARISON SUMMARY (mu0={mu0})\n{'='*70}")
    print(f"{'Size':<10} {'Grid':<8} {'Pivotal':<10} {'Edges':<8} {'Final Loss':<12}")
    print("-" * 70)
    for name, res in results.items():
        g = res['grid_size']
        print(f"{name:<10} {g}x{g:<6} {len(res['pivotal_states']):<10} {len(res['world_graph'].edges):<8} {res['final_loss']:<12.4f}")

    return results


def main():
    """
    test_phase1_with_diagnostics(config={
        'maze_size': externalconfig['maze_size'],
        'phase1_iterations': externalconfig['phase1_iterations'],
        'vae_mu0': externalconfig['vae_mu0'],
        'device': externalconfig['device'],
    })
    """
    train_full_phase1_phase2()       # Phase 1 + Phase 2 together (saves checkpoint automatically)
    #run_phase2_standalone('phase1_checkpoint_MEDIUM.pt', config_overrides=externalconfig, fixed_balls=True, phase2_animation=True)  # fixed_balls=False for random
    #render_phase2_episode_gif('phase1_checkpoint_MEDIUM.pt', filename='phase2_final_episode.mp4', fps=15, max_steps=500)
    # run_phase1_comparison()
    # run_phase1_size_comparison()


if __name__ == "__main__":
    main()
