import random
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# minigrid imports
from minigrid.core.world_object import Wall

# Local imports
from wrappers.minigrid_wrapper import MinigridWrapper, EnvModes, EnvSizes
from utils.statistics_buffer import StatBuffer
from local_networks.vaesystem import VAESystem
from local_networks.policy_networks import GoalConditionedPolicy
from utils.misc import manhattan_distance, resolve_device, _walk_away_from_spawn
from utils.checkpoint import save_phase1_checkpoint, load_phase1_checkpoint, restore_maze_from_grid_state
from utils.graph_manager import GraphManager
from utils.visualization import (plot_training_diagnostics, save_graph_visualization,
                                  render_phase3_episode_gif,
                                  _run_and_save_episode, print_grid_image, _plot_phase1_run,
                                  plot_worker_pretrain_diagnostics,
                                  plot_manager_wide_pretrain_diagnostics,
                                  plot_manager_narrow_pretrain_diagnostics)
from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker, HierarchicalTrainer
from bufferclasses import NarrowReplayBuffer, WorkerEpisodeReplayBuffer
from config import externalconfig

#----------------------------------------------------------------------------#
#                          DIAGNOSTIC FUNCTIONS                              #
#----------------------------------------------------------------------------#

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
    valid_cells_diag = {
        (x, y)
        for x in range(env.width)
        for y in range(env.height)
        if env._is_traversable(env.grid.get(x, y))
    }
    worker.valid_cells = valid_cells_diag
    worker.build_wall_mask(env.width, env.height)

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
        wide_goal, narrow_goal, _, _, _ = manager.get_manager_action(current_pos, valid_cells=valid_cells_diag)
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
    
    policy = GoalConditionedPolicy(lr=config['goal_policy_lr'], maze_size=config['maze_size'].value, device=config['device'])
    vae_system = VAESystem(
        state_dim=16,
        action_vocab_size=7, 
        mu0=config['vae_mu0'], 
        grid_size=env.size,
        device=config['device']
    )
    buffer = StatBuffer()
    
    # Run alternating training (now with persistent KL)
    pivotal_states, world_graph, loop_metrics = alternating_training_loop(
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
    Expects results from test_phase1_with_diagnostics() — not from run_phase1_mu0_sweep()."""
    
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

def print_phase1_results_table(runs_dict):
    """Print a comparison table across multiple Phase 1 runs.
    Expects {name: results} where each results comes from test_phase1_with_diagnostics().
    Do NOT pass the output of run_phase1_mu0_sweep() — it uses a different metrics structure."""
    
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

#----------------------------------------------------------------------------#
#                      PHASE 1: WORLD GRAPH DISCOVERY                        #
#----------------------------------------------------------------------------#

def alternating_training_loop(env, policy, vae_system, buffer, max_iterations: int = 8, convergence_threshold: float = 0.01,
                              explore_top_fraction: float = 0.20,
                              diversity_walk_number: int = 10,
                              walk_length: int = 250,
                              walk_bias: float = 0.65,
                              walk_episodes: int = 4,
                              graph_walk_length: int = 20,
                              graph_num_attempts: int = 70,
                              spread_alpha: float = 0.0,
                              skip_gcp_refine: bool = False):
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

    # Bootstrap buffer before the training loop so no iteration is wasted
    if buffer.episodes_in_buffer < 3:
        print("Bootstrapping buffer before training...")
        for _ in range(5):
            env.reset()
            start_pos = tuple(env.agent_pos)
            episodes = policy.collect_episodes_from_position(
                env, start_pos, num_episodes=6, max_episode_length=100, vae_system=vae_system
            )
            if episodes:
                buffer.add_episodes(episodes)


    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        print(f"Current KL weight: {persistent_kl_weight:.3f}")

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
            current_recon_loss = vae_system.training_history[-1]['reconstruction_loss']
            reconstruction_losses.append(current_recon_loss)
            print(f"Current reconstruction loss: {current_recon_loss:.4f}")
        
        metrics['num_pivotal_states_per_iteration'].append(len(pivotal_states))

        print(f"Discovered {len(pivotal_states)} pivotal states: {pivotal_states[:3]}...")
        
        # Collect trajectories from pivotal states
        # Sort by distance from spawn descending — farthest states first — to break the
        # self-reinforcing clustering loop that keeps all pivotal states near spawn.

        coverage_threshold = 50
        sorted_by_dist = sorted(pivotal_states, key=lambda s: manhattan_distance(s, spawn_pos), reverse=True)

        if len(pivotal_states) < coverage_threshold:
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
            
            #Curiosity Weight Scheduling
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
        curiosity_weight_for_diversity_walk = 0.0 if iteration == 0 else max(0.15, 0.5 - (iteration * 0.05))
        print(f"Spatial diversity: {diversity_walk_number} biased walks from spawn {spawn_pos}...")
        for _wi in range(diversity_walk_number):
            _dst = _walk_away_from_spawn(env, spawn_pos, walk_length=walk_length, bias=walk_bias)
            dist_from_spawn = manhattan_distance(_dst, spawn_pos)
            print(f"  Walk {_wi+1}: reached {_dst} (dist={dist_from_spawn})")
            try:
                _eps = policy.collect_episodes_from_position(
                    env, _dst, num_episodes=walk_episodes, max_episode_length=100,
                    vae_system=vae_system, curiosity_weight=curiosity_weight_for_diversity_walk
                )
                if _eps:
                    buffer.add_episodes(_eps)
            except Exception as e:
                print(f"  Walk {_wi+1} collection failed: {e}")

        # Check for convergence (not before min_iterations to ensure coverage)
        min_iterations_before_convergence = 5
        if len(reconstruction_losses) >= 3 and iteration + 1 >= min_iterations_before_convergence:
            recent_losses = reconstruction_losses[-3:]
            loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
            avg_change = sum(loss_changes) / len(loss_changes)

            print(f"Average loss change over last 3 iterations: {avg_change:.5f}")

            if avg_change < convergence_threshold:
                print("Reconstruction loss has plateaued - training converged!")
                break
    

    # Ensure spawn point is a pivotal state so Phase 3 always starts on the graph
    spawn = tuple(env.agent_start_pos)
    if spawn not in pivotal_states:
        pivotal_states.append(spawn)
        print(f"[Phase 1] Spawn {spawn} not in pivotal states — added (total: {len(pivotal_states)})")

    # Construct world graph
    world_graph = policy.complete_world_graph_discovery(env, pivotal_states,
                                                         graph_walk_length=graph_walk_length,
                                                         graph_num_attempts=graph_num_attempts,
                                                         skip_gcp_refine=skip_gcp_refine)
    
    # Final summary
    print(f"\nAlternating Training Complete!")
    print(f"Total iterations: {len(reconstruction_losses)}")
    print(f"Final episodes in buffer: {buffer.episodes_in_buffer}")
    print(f"Final pivotal states ({len(pivotal_states)}): {pivotal_states}")
    
    return pivotal_states, world_graph, metrics

def run_phase1_mu0_sweep(mu0_values=None, maze_size=EnvSizes.MEDIUM, iterations=50):
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

        policy = GoalConditionedPolicy(lr=externalconfig['goal_policy_lr'], maze_size=externalconfig['maze_size'].value, device=device)
        vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=mu0, grid_size=base_env.size, device=device)
        buffer = StatBuffer()
        base_env.phase = 1

        pivotal_states, world_graph, metrics = alternating_training_loop(
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

def run_phase1_size_sweep(sizes=None, mu0=9.0, iterations=50):
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

        policy = GoalConditionedPolicy(lr=externalconfig['goal_policy_lr'], maze_size=externalconfig['maze_size'].value, device=device)
        vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=mu0, grid_size=env.size, device=device)
        buffer = StatBuffer()

        pivotal_states, world_graph, metrics = alternating_training_loop(
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

#----------------------------------------------------------------------------#
#                           PHASE 2: PRETRAIN                                #
#----------------------------------------------------------------------------#

def run_worker_pretrain(env, worker, grid_state, config, device):
    """
    Curriculum pre-training of Worker/GCP on short-range navigation.
    Curriculum keys (threshold_rN, repeat_rN, i/f/dropby_rN, substage_cap_rN)
    are read directly from config.
    Within each r, max_steps compresses each time `repeat` consecutive eval windows all
    exceed the threshold. Sequence is i_steps → i_steps-dropby → ... → f_steps (clamped).
    """
    import torch.nn.functional as F
    from torch.distributions import Categorical

    env.phase = 1
    env.randomgen = False
    env.reset()
    restore_maze_from_grid_state(env, grid_state)
    env.reset()

    valid_cells = [
        (x, y)
        for x in range(1, env.width - 1)
        for y in range(1, env.height - 1)
        if env._is_traversable(env.grid.get(x, y))
    ]
    valid_set = set(valid_cells)
    worker.valid_cells = valid_set
    worker.build_wall_mask(env.width, env.height)

    def _build_steps_seq(i_steps, dropby, f_steps):
        steps, cur = [], i_steps
        while True:
            steps.append(cur)
            if cur <= f_steps:
                break
            cur = max(cur - dropby, f_steps)
        return steps

    configuration = config
    default_cap = 5000
    curriculum = [
        (
            r,
            configuration.get(f'threshold_r{r}', thresh_def),
            configuration.get(f'repeat_r{r}', 1),
            _build_steps_seq(
                configuration.get(f'i_steps_r{r}', i_def),
                configuration.get(f'dropby_r{r}',  drop_def),
                configuration.get(f'f_steps_r{r}', f_def),
            ),
            configuration.get(f'substage_cap_r{r}', default_cap),
        )
        for r, thresh_def, i_def, drop_def, f_def in [
            (1, 0.95, 25,  5,  10),
            (2, 0.90, 80,  10, 30),
            (3, 0.90, 200, 40, 40),
        ]
    ]
    eval_window = 50

    # No normalization on goal delta throughout pretrain: raw cells give a strong gradient
    # signal and Tanh handles values up to 3 without saturation (tanh(0.4*3)≈0.74).
    # The critical invariant is that the divisor stays CONSTANT across all r-stages so
    # the network doesn't have to relearn what the same input value means.
    worker.goal_norm_div = 1.0

    print(f"\n{'='*70}")
    print(f"INTERMEDIATE PHASE: Worker/GCP Curriculum Pre-training")
    print(f"  {len(valid_cells)} valid cells | eval window: {eval_window} episodes")
    for r, threshold, repeat, steps_seq, substage_cap in curriculum:
        print(f"  r={r}: threshold={threshold*100:.0f}% x{repeat}, steps={steps_seq}, cap={substage_cap}")
    print(f"{'='*70}")

    goal_reached_history  = []
    episode_reward_history = []
    episode_length_history = []
    training_metrics      = []
    total_episode         = 0

    ppo_epochs = config.get('ppo_epochs', 4)
    clip_eps   = config.get('ppo_clip_eps', 0.2)
    gae_lambda = config.get('gae_lambda', 0.95)

    use_per         = config.get('worker_per_use', False)
    per_buffer_size = config.get('worker_per_buffer_size', 500)
    per_warmup      = config.get('worker_per_warmup', 250)
    per_replay_freq = config.get('worker_per_replay_freq', 250)
    per_batch_eps   = config.get('worker_per_batch_episodes', 50)
    per_alpha       = config.get('worker_per_alpha', 0.6)
    per_beta_start  = config.get('worker_per_beta_start', 0.4)
    per_lr_factor   = config.get('worker_per_lr_factor', 0.5)
    if use_per:
        per_replay_buffer = WorkerEpisodeReplayBuffer(per_buffer_size, alpha=per_alpha)

    for r, threshold, repeat, steps_seq, substage_cap in curriculum:
        # Reset Adam momentum accumulated from previous r-stage distribution
        worker.optimizer.state.clear()
        print(f"\n  [r={r}] optimizer momentum reset | goal_norm_div={worker.goal_norm_div}")
        base_entropy_coef = worker.entropy_coef
        base_lr           = worker.optimizer.param_groups[0]['lr']
        lr_ramp_ep        = 3 * eval_window   # 150 ep up → peak 1.5x, 150 ep down → base
        r_stage_ep = 0  # tracks episodes within this r-stage for entropy/LR decay
        if use_per:
            per_replay_buffer.clear()

        for max_steps in steps_seq:
            # Reset Adam momentum at every step-stage: when max_steps shrinks the return
            # distribution shifts (smaller scale), so stale momentum corrupts advantage estimates
            # and causes the value-loss spike / policy collapse pattern.
            worker.optimizer.state.clear()

            # Target ~30 transitions per PPO call to prevent critic overfitting on tiny batches.
            # With max_steps=5 this batches 6 episodes; at max_steps=30 it's 1 (no change).
            batch_eps = max(1, math.ceil(30 / max_steps))
            print(f"\n  Stage r={r}, max_steps={max_steps} "
                  f"(threshold={threshold*100:.0f}%, repeat={repeat}, "
                  f"advance after {repeat*eval_window} eps, batch_eps={batch_eps})")
            substage_ep        = 0
            consecutive_passes = 0
            next_check         = eval_window   # next substage_ep at which to evaluate

            # Batch accumulators — flushed every batch_eps episodes or on stage advance
            batch_states_acc, batch_actions_acc, batch_rewards_acc = [], [], []
            batch_values_acc, batch_log_probs_acc                  = [], []
            batch_next_states_acc, batch_dones_acc                 = [], []
            episodes_in_batch = 0

            def _flush_batch():
                nonlocal episodes_in_batch
                nonlocal batch_states_acc, batch_actions_acc, batch_rewards_acc
                nonlocal batch_values_acc, batch_log_probs_acc
                nonlocal batch_next_states_acc, batch_dones_acc
                if not batch_states_acc:
                    return
                metrics = worker.update_policy(
                    batch_states_acc, batch_actions_acc, batch_rewards_acc,
                    batch_values_acc, batch_log_probs_acc,
                    next_states=batch_next_states_acc, dones=batch_dones_acc,
                    ppo_epochs=ppo_epochs, clip_eps=clip_eps, gae_lambda=gae_lambda,
                )
                if metrics:
                    n = len(batch_actions_acc)
                    metrics['frac_left']    = batch_actions_acc.count(0) / n
                    metrics['frac_right']   = batch_actions_acc.count(1) / n
                    metrics['frac_forward'] = batch_actions_acc.count(2) / n
                    training_metrics.append(metrics)
                batch_states_acc, batch_actions_acc, batch_rewards_acc = [], [], []
                batch_values_acc, batch_log_probs_acc                  = [], []
                batch_next_states_acc, batch_dones_acc                 = [], []
                episodes_in_batch = 0

            while substage_ep < substage_cap:
                spawn = random.choice(valid_cells)
                env.agent_start_pos = spawn
                env.agent_start_dir = random.randint(0, 3)
                env.reset()
                state = tuple(env.agent_pos)

                # Manhattan ball of radius r
                candidates = [
                    (state[0] + dx, state[1] + dy)
                    for dx in range(-r, r + 1)
                    for dy in range(-r, r + 1)
                    if 0 < abs(dx) + abs(dy) <= r
                    and (state[0] + dx, state[1] + dy) in valid_set
                ]
                if not candidates:
                    goal_reached_history.append(0.0)
                    episode_reward_history.append(0.0)
                    total_episode += 1
                    substage_ep   += 1
                    continue

                narrow_goal = random.choice(candidates)
                worker.reset_worker_state()

                worker_states, worker_actions, worker_rewards, worker_values, worker_log_probs = [], [], [], [], []
                worker_next_states, worker_dones = [], []
                goal_reached = False

                for step in range(max_steps):
                    agent_dir = env.agent_dir
                    action_logits, value = worker.forward(state, agent_dir, narrow_goal)
                    probs = F.softmax(action_logits, dim=0)
                    dist = Categorical(probs)
                    idx = dist.sample()
                    log_prob = dist.log_prob(idx)
                    action = idx.item()

                    prev_dist = abs(state[0] - narrow_goal[0]) + abs(state[1] - narrow_goal[1])

                    try:
                        _, _, terminated, truncated, _ = env.step(action)
                        next_state    = tuple(env.agent_pos)
                        next_agent_dir = env.agent_dir
                    except (AssertionError, IndexError):
                        next_state    = state
                        next_agent_dir = agent_dir
                        terminated    = False
                        truncated     = False

                    curr_dist = abs(next_state[0] - narrow_goal[0]) + abs(next_state[1] - narrow_goal[1])
                    done   = (next_state == narrow_goal) or terminated or truncated
                    reward = 1.0 if next_state == narrow_goal else (prev_dist - curr_dist) * 0.1 - 0.01

                    worker_states.append((state, agent_dir, narrow_goal))
                    worker_actions.append(action)
                    worker_rewards.append(reward)
                    worker_values.append(value.squeeze())
                    worker_log_probs.append(log_prob)
                    worker_next_states.append((next_state, next_agent_dir, narrow_goal))
                    worker_dones.append(done)

                    state = next_state
                    if next_state == narrow_goal:
                        goal_reached = True
                        break
                    if terminated or truncated:
                        break

                # Force done=True on timeout so GAE doesn't bootstrap across episode boundary
                if worker_dones and not worker_dones[-1]:
                    worker_dones[-1] = True

                goal_reached_history.append(float(goal_reached))
                episode_reward_history.append(sum(worker_rewards) if worker_rewards else 0.0)
                episode_length_history.append(len(worker_rewards))
                total_episode += 1
                substage_ep   += 1

                # Entropy annealing: base → base*0.3 linearly over the substage budget.
                # Resets at every step-stage so each new difficulty level starts with
                # full entropy, then converges progressively as the episode cap is used up.
                t = substage_ep / max(substage_cap, 1)
                worker.entropy_coef = base_entropy_coef * (1.0 - 0.7 * t)

                # LR triangular warm-up: ramp 1x→1.5x over lr_ramp_ep, then 1.5x→1x
                if r_stage_ep < lr_ramp_ep:
                    lr_factor = 1.0 + 0.5 * (r_stage_ep / lr_ramp_ep)
                elif r_stage_ep < 2 * lr_ramp_ep:
                    lr_factor = 1.5 - 0.5 * ((r_stage_ep - lr_ramp_ep) / lr_ramp_ep)
                else:
                    lr_factor = 1.0
                for pg in worker.optimizer.param_groups:
                    pg['lr'] = base_lr * lr_factor

                r_stage_ep += 1

                # Accumulate episode into batch
                if worker_rewards:
                    batch_states_acc.extend(worker_states)
                    batch_actions_acc.extend(worker_actions)
                    batch_rewards_acc.extend(worker_rewards)
                    batch_values_acc.extend(worker_values)
                    batch_log_probs_acc.extend(worker_log_probs)
                    batch_next_states_acc.extend(worker_next_states)
                    batch_dones_acc.extend(worker_dones)
                    episodes_in_batch += 1

                if use_per and worker_rewards:
                    per_replay_buffer.add(
                        (worker_states, worker_actions, worker_rewards,
                         [lp.detach() for lp in worker_log_probs],
                         worker_next_states, worker_dones),
                        total_reward=sum(worker_rewards),
                    )

                if total_episode % 50 == 0:
                    recent     = goal_reached_history[-eval_window:]
                    recent_ach = sum(recent) / len(recent)
                    m = training_metrics[-1] if training_metrics else {}
                    print(f"    Ep {total_episode:>5} | Ach: {recent_ach*100:.1f}% "
                          f"[{consecutive_passes}/{repeat}] | "
                          f"PL: {m.get('policy_loss',0):.3f} | VL: {m.get('value_loss',0):.3f} | "
                          f"Ent: {m.get('entropy',0):.3f} | GN: {m.get('grad_norm',0):.3f}")

                # Check threshold every eval_window episodes; require `repeat` consecutive passes
                should_advance = False
                if substage_ep >= next_check:
                    next_check += eval_window
                    recent_ach  = sum(goal_reached_history[-eval_window:]) / eval_window
                    if recent_ach >= threshold:
                        consecutive_passes += 1
                        print(f"    [+] window ach={recent_ach*100:.1f}% "
                              f"consecutive={consecutive_passes}/{repeat}")
                        if consecutive_passes >= repeat:
                            should_advance = True
                    else:
                        consecutive_passes = 0

                # Fire PPO update when batch is full or we are about to advance stage
                if (episodes_in_batch >= batch_eps or should_advance) and batch_states_acc:
                    _flush_batch()

                if (use_per
                        and r_stage_ep >= per_warmup
                        and (r_stage_ep - per_warmup) % per_replay_freq == 0
                        and len(per_replay_buffer) >= per_batch_eps):
                    beta = per_beta_start + (1.0 - per_beta_start) * min(
                        1.0, r_stage_ep / max(1, per_warmup + per_replay_freq * 10))
                    rep_episodes, _ = per_replay_buffer.sample(per_batch_eps, beta)
                    for pg in worker.optimizer.param_groups:
                        pg['_saved_lr'] = pg['lr']
                        pg['lr'] *= per_lr_factor
                    rep_states, rep_actions, rep_rewards = [], [], []
                    rep_log_probs, rep_next_states, rep_dones, rep_values = [], [], [], []
                    with torch.no_grad():
                        for ep_data in rep_episodes:
                            s_list, a_list, r_list, lp_list, ns_list, d_list = ep_data
                            for (s, ad, ng) in s_list:
                                _, v = worker.forward(s, ad, ng)
                                rep_values.append(v.squeeze())
                            rep_states.extend(s_list)
                            rep_actions.extend(a_list)
                            rep_rewards.extend(r_list)
                            rep_log_probs.extend(lp_list)
                            rep_next_states.extend(ns_list)
                            rep_dones.extend(d_list)
                    if rep_rewards:
                        worker.update_policy(
                            rep_states, rep_actions, rep_rewards,
                            rep_values, rep_log_probs,
                            next_states=rep_next_states, dones=rep_dones,
                            ppo_epochs=ppo_epochs, clip_eps=clip_eps, gae_lambda=gae_lambda,
                        )
                        print(f"    [PER] r_stage_ep={r_stage_ep} | "
                              f"replayed {len(rep_episodes)} eps / {len(rep_rewards)} steps")
                    for pg in worker.optimizer.param_groups:
                        pg['lr'] = pg['_saved_lr']

                if should_advance:
                    print(f"    -> Threshold met {repeat}x. Advancing.")
                    break

    final_ach = sum(goal_reached_history[-100:]) / min(100, len(goal_reached_history)) * 100
    avg_len = sum(episode_length_history) / len(episode_length_history) if episode_length_history else 0
    print(f"\nWorker curriculum complete. Final achievement (last 100 ep): {final_ach:.1f}%")
    print(f"Total episodes: {total_episode} | Avg episode length: {avg_len:.1f}")
    plot_worker_pretrain_diagnostics(goal_reached_history, episode_reward_history,
                                     episode_length_history, training_metrics)
    return goal_reached_history

def refine_edges_with_worker(env, worker, world_graph, pivotal_states, config, device):
    """
    Refine world graph edges using the pre-trained Worker for navigation.
    Called after run_worker_pretrain. Adds new edges and replaces existing ones
    with shorter paths found by the Worker (replaces GCP refine_paths_with_goal_policy).
    """
    graph_attempts  = config.get('worker_graph_attempts', 50)
    graph_max_steps = config.get('worker_graph_max_steps', 100)

    pivotal_set = set(tuple(p) for p in pivotal_states)
    found_edges = {}  # (start, end) -> shortest path found so far

    print(f"\n{'='*70}")
    print(f"POST-PRETRAIN EDGE REFINE (Worker) | {len(pivotal_states)} pivots | "
          f"{graph_attempts} attempts/pivot | max {graph_max_steps} steps/attempt")

    env.phase     = 1
    env.randomgen = False

    for start in pivotal_states:
        candidates = [p for p in pivotal_states if p != start]
        if not candidates:
            continue
        for _ in range(graph_attempts):
            target = random.choice(candidates)
            env.agent_start_pos = start
            env.agent_start_dir = random.randint(0, 3)
            env.reset()
            state = tuple(env.agent_pos)
            path  = [state]
            worker.reset_worker_state()

            for _ in range(graph_max_steps):
                agent_dir = env.agent_dir
                with torch.no_grad():
                    action_logits, _ = worker.forward(state, agent_dir, target)
                action = torch.argmax(F.softmax(action_logits, dim=0)).item()
                try:
                    _, _, terminated, truncated, _ = env.step(action)
                    next_state = tuple(env.agent_pos)
                except (AssertionError, IndexError):
                    break
                path.append(next_state)
                if next_state in pivotal_set and next_state != tuple(start):
                    key = (tuple(start), next_state)
                    if key not in found_edges or len(path) < len(found_edges[key]):
                        found_edges[key] = list(path)
                state = next_state
                if terminated or truncated:
                    break

    edges_added   = 0
    edges_improved = 0
    for (src, dst), path in found_edges.items():
        new_weight = len(path) - 1
        existing   = world_graph.edges.get((src, dst))
        if existing is None:
            world_graph.add_edge(src, dst, new_weight, path)
            edges_added += 1
        elif new_weight < existing['weight']:
            world_graph.add_edge(src, dst, new_weight, path)
            edges_improved += 1

    avg_reach = (sum(len(world_graph.get_reachable_nodes(p)) for p in pivotal_states)
                 / max(1, len(pivotal_states)))
    print(f"  New edges: {edges_added} | Improved: {edges_improved} | "
          f"Total: {len(world_graph.edges)} | Avg reachable: {avg_reach:.1f}/{len(pivotal_states)}")
    print(f"{'='*70}")
    return world_graph

def run_manager_wide_pretrain(env, manager, grid_state, config, device):
    """Intermediate phase: pre-train Manager wide head on ball-proximity goal selection."""
    horizons_per_ep = config.get('manager_wide_horizons_per_episode', 20)
    r               = config.get('neighborhood_size', 3)
    base_lr         = config.get('manager_lr', 5e-4)
    ppo_epochs      = config.get('manager_wide_ppo_epochs', 4)
    ppo_batch_size  = config.get('manager_wide_ppo_batch_size', 1)
    episodes        = config.get('manager_wide_pretrain_episodes', 0)
    pretrain_r      = r + config.get('manager_wide_pretrain_r_offset', 0)
    lr_start        = base_lr
    lr_end          = base_lr * config.get('manager_wide_lr_end_factor', 1.0)

    env.phase = 2
    env.fixed_ball_positions = None

    env.randomgen = False
    env.reset()
    restore_maze_from_grid_state(env, grid_state)
    env.randomgen = True
    env.firstgen  = False
    env.reset()

    valid_cells = [
        (x, y)
        for x in range(1, env.width - 1)
        for y in range(1, env.height - 1)
        if env._is_traversable(env.grid.get(x, y))
    ]
    maze_diagonal = (env.width - 2) + (env.height - 2)
    if valid_cells:
        env.agent_start_pos = random.choice(valid_cells)

    print(f"\n{'='*70}")
    print(f"INTERMEDIATE PHASE: Manager Wide Goal Pre-training")
    print(f"  {episodes} ep | {horizons_per_ep} horizons/ep | pretrain_r={pretrain_r} | "
          f"LR {lr_start:.1e}->{lr_end:.1e} cosine | diagonal={maze_diagonal}")
    print(f"{'='*70}")

    ball_coverage_history = []
    avg_reward_history    = []
    avg_dist_history      = []
    rollout_buffer        = []

    manager.optimizer = torch.optim.Adam(manager.parameters(), lr=lr_start)

    for ep in range(episodes):
        # ── Cosine LR annealing ──────────────────────────────────────────
        cosine_factor = 0.5 * (1 + math.cos(math.pi * ep / max(1, episodes)))
        current_lr    = lr_end + (lr_start - lr_end) * cosine_factor
        for pg in manager.optimizer.param_groups:
            pg['lr'] = current_lr

        # ── Episode rollout ──────────────────────────────────────────────
        state              = random.choice(valid_cells)
        active_balls       = env.reset_for_wide_pretrain(valid_cells, config.get('num_balls', 5))
        initial_ball_count = len(active_balls)

        manager.reset_manager_state()

        m_states, m_wide, m_narrow = [], [], []
        m_rewards, m_values, m_log_probs, m_entropies = [], [], [], []
        m_dists     = []
        m_balls     = []
        m_wide_idxs = []

        for h in range(horizons_per_ep):
            if not active_balls:
                break

            m_balls.append(list(active_balls))

            wide_logits, _, value = manager.forward(state, list(active_balls))
            wide_probs = F.softmax(wide_logits, dim=0)
            wide_dist  = torch.distributions.Categorical(wide_probs)
            wide_idx   = wide_dist.sample()
            log_prob   = wide_dist.log_prob(wide_idx)
            entropy    = -(wide_probs * torch.log(wide_probs + 1e-8)).sum().detach()
            wide_goal  = manager.pivotal_states[wide_idx.item()]
            manager.prev_wide_goal = wide_goal
            narrow_goal = wide_goal
            value = value.squeeze()
            if manager.hidden_state is not None:
                manager.hidden_state = tuple(hs.detach() for hs in manager.hidden_state)

            m_wide_idxs.append(wide_idx.item())

            covered = [b for b in active_balls if manhattan_distance(wide_goal, b) <= pretrain_r]
            if covered:
                closest         = min(covered, key=lambda b: manhattan_distance(wide_goal, b))
                dist_to_closest = manhattan_distance(wide_goal, closest)
                reward          = 1.0 + 2.0 * (pretrain_r - dist_to_closest) / pretrain_r
                active_balls.discard(closest)
            else:
                dist_to_closest = min(manhattan_distance(wide_goal, b) for b in active_balls)
                reward          = -dist_to_closest / maze_diagonal

            m_dists.append(dist_to_closest)
            m_states.append(state)
            m_wide.append(wide_goal)
            m_narrow.append(narrow_goal)
            m_rewards.append(reward)
            m_values.append(value)
            m_log_probs.append(log_prob)
            m_entropies.append(entropy.detach())

            state = wide_goal

        ball_coverage_history.append((initial_ball_count - len(active_balls)) / max(1, initial_ball_count))
        avg_reward_history.append(sum(m_rewards) / len(m_rewards) if m_rewards else 0.0)
        avg_dist_history.append(sum(m_dists)    / len(m_dists)    if m_dists    else 0.0)

        if len(m_rewards) > 1:
            rollout = (m_states, m_wide, m_narrow, m_rewards, m_values,
                       m_log_probs, m_entropies, m_balls, m_wide_idxs)
            rollout_buffer.append(rollout)
            if len(rollout_buffer) >= ppo_batch_size:
                manager.update_policy_batched(
                    rollout_buffer, ppo_epochs=ppo_epochs,
                    entropy_coef_override=manager.entropy_coef,
                )
                rollout_buffer = []

        if (ep + 1) % 10000 == 0:
            recent      = ball_coverage_history[-10000:]
            recent_dist = avg_dist_history[-10000:]
            recent_rew  = avg_reward_history[-10000:]
            avg_entropy = sum(m_entropies).item() / len(m_entropies) if m_entropies else 0.0
            num_pivots  = len(manager.pivotal_states)
            max_entropy = math.log(num_pivots) if num_pivots > 1 else 1.0
            print(f"  Ep {ep+1:>6}/{episodes} "
                  f"[r={pretrain_r} lr={current_lr:.1e}] | "
                  f"Coverage: {sum(recent)/len(recent)*100:.1f}% | "
                  f"AvgDist: {sum(recent_dist)/len(recent_dist):.2f} | "
                  f"AvgRew: {sum(recent_rew)/len(recent_rew):+.3f} | "
                  f"Entropy: {avg_entropy:.3f}/{max_entropy:.3f} "
                  f"({100*avg_entropy/max_entropy:.0f}%)")

    if rollout_buffer:
        manager.update_policy_batched(
            rollout_buffer, ppo_epochs=ppo_epochs,
            entropy_coef_override=manager.entropy_coef,
        )

    final_cov  = sum(ball_coverage_history[-100:]) / min(100, len(ball_coverage_history)) * 100
    final_dist = sum(avg_dist_history[-100:])      / min(100, len(avg_dist_history))
    print(f"\nManager Wide Pre-training complete. "
          f"Final coverage: {final_cov:.1f}% | AvgDist: {final_dist:.2f} (last 100 ep)")
    plot_manager_wide_pretrain_diagnostics(ball_coverage_history, avg_reward_history, avg_dist_history)
    return ball_coverage_history

def run_manager_narrow_pretrain(env, manager, grid_state, config, device):
    """
    Intermediate phase: pre-train Manager narrow head on precise ball targeting.
    No Worker, no traversal. Oracle wide_goal = closest pivotal state to a random ball,
    guaranteeing a ball is reachable from the neighborhood. Narrow head learns to select
    the neighborhood cell closest to the ball.
    Reward: +2 on ball (dist=0), +0.1 adjacent (dist=1), -1 otherwise.
    """
    episodes       = config.get('manager_narrow_pretrain_episodes', 0)
    horizons_per_ep = config.get('manager_narrow_horizons_per_episode', 20)
    r              = config.get('neighborhood_size', 3)

    env.phase = 2
    env.fixed_ball_positions = None

    # Restore Phase 1 maze first so valid_cells is computed from clean grid (no balls)
    env.randomgen = False
    # Set a safe start pos from grid_state before reset (current grid may have balls on agent_start_pos)
    temp_valid = [(x, y) for x in range(1, env.width - 1) for y in range(1, env.height - 1)
                  if grid_state[y][x] != '#']
    if temp_valid:
        env.agent_start_pos = random.choice(temp_valid)
    env.reset()
    restore_maze_from_grid_state(env, grid_state)

    valid_cells = [
        (x, y)
        for x in range(1, env.width - 1)
        for y in range(1, env.height - 1)
        if env._is_traversable(env.grid.get(x, y))
    ]
    valid_set = set(valid_cells)
    manager._valid_cells = valid_set

    if valid_cells:
        env.agent_start_pos = random.choice(valid_cells)
    env.randomgen = True
    env.firstgen = False
    env.reset()

    print(f"\n{'='*70}")
    print(f"INTERMEDIATE PHASE: Manager Narrow Goal Pre-training")
    print(f"  {episodes} episodes | {horizons_per_ep} horizons/ep | r={r}")
    print(f"{'='*70}")

    ppo_epochs = config.get('manager_narrow_ppo_epochs', 4)

    entropy_start = config.get('manager_narrow_entropy_start', 0.3)
    entropy_end   = config.get('manager_narrow_entropy_end', 0.001)
    _original_entropy_coef = manager.entropy_coef

    use_per             = config.get('manager_narrow_use_per', False)
    per_warmup          = config.get('manager_narrow_per_warmup', 1000)
    per_replay_freq     = config.get('manager_narrow_per_replay_freq', 250)
    per_buffer_size     = config.get('manager_narrow_per_buffer_size', 5000)
    per_batch_size      = config.get('manager_narrow_per_batch_size', 32)
    per_alpha           = config.get('manager_narrow_per_alpha', 0.6)
    per_beta_start      = config.get('manager_narrow_per_beta_start', 0.4)
    per_lr_factor       = config.get('manager_narrow_per_lr_factor', 0.1)
    per_entropy_coef    = config.get('manager_narrow_per_entropy_coef', 0.05)
    if use_per:
        replay_buffer = NarrowReplayBuffer(per_buffer_size, alpha=per_alpha)

    hit_history        = []   # fraction of horizons with narrow_goal on ball (dist=0)
    near_history       = []   # fraction of horizons with dist<=1
    avg_reward_history = []

    for episode in range(episodes):
        # Cosine entropy annealing: high → low to prevent early collapse
        t = episode / max(1, episodes - 1)
        manager.entropy_coef = entropy_end + 0.5 * (entropy_start - entropy_end) * (1 + math.cos(math.pi * t))

        spawn = random.choice(valid_cells)
        env.agent_start_pos = spawn
        env.agent_start_dir = random.randint(0, 3)
        env.reset()

        active_balls = list(env.active_balls)
        if not active_balls:
            hit_history.append(0.0)
            near_history.append(0.0)
            avg_reward_history.append(0.0)
            continue

        manager.reset_manager_state()

        m_states, m_wide, m_narrow = [], [], []
        m_rewards, m_values, m_log_probs, m_entropies, m_balls = [], [], [], [], []
        m_dx, m_dy = [], []
        hits  = 0
        nears = 0

        for h in range(horizons_per_ep):
            # Oracle: random ball → closest pivotal state as wide_goal
            target_ball = random.choice(active_balls)
            wide_goal   = min(manager.pivotal_states,
                              key=lambda p: manhattan_distance(p, target_ball))
            state = wide_goal  # agent is at oracle wide_goal

            # Skip irresolvable samples: ball outside neighborhood → hit impossible
            if manhattan_distance(wide_goal, target_ball) > r:
                continue

            # Value from LSTM (fresh per horizon — no accumulated sequence noise)
            manager.hidden_state = None
            _, _, value = manager.forward(state, active_balls)
            value = value.squeeze()

            # Narrow policy: purely geometric [dx, dy] relative to wide_goal
            dx = float(target_ball[0] - wide_goal[0])
            dy = float(target_ball[1] - wide_goal[1])
            narrow_logits = manager.narrow_head(torch.tensor([dx, dy], dtype=torch.float32, device=device))

            neighborhood_full = manager.get_neighborhood(wide_goal)
            valid_indices = [i for i, cell in enumerate(neighborhood_full) if cell in valid_set]
            if not valid_indices:
                continue

            idx_t        = torch.tensor(valid_indices, dtype=torch.long, device=device)
            valid_logits = narrow_logits[idx_t]
            valid_probs  = F.softmax(valid_logits, dim=0)
            valid_dist   = torch.distributions.Categorical(valid_probs)
            local_idx    = valid_dist.sample()
            log_prob     = valid_dist.log_prob(local_idx)
            entropy      = -(valid_probs * torch.log(valid_probs + 1e-8)).sum().detach()
            narrow_goal  = neighborhood_full[valid_indices[local_idx.item()]]

            dist_to_ball = manhattan_distance(narrow_goal, target_ball)
            if dist_to_ball == 0:
                reward = 3.0
                hits  += 1
                nears += 1
            elif dist_to_ball == 1:
                reward = 0.5
                nears += 1
            elif dist_to_ball == 2:
                reward = -1.0
            else:
                reward = -3.0

            m_states.append(state)
            m_wide.append(wide_goal)
            m_narrow.append(narrow_goal)
            m_rewards.append(reward)
            m_values.append(value)
            m_log_probs.append(log_prob)
            m_entropies.append(entropy)
            m_balls.append(list(active_balls))
            m_dx.append(dx)
            m_dy.append(dy)

        total_h = max(1, len(m_rewards))
        hit_history.append(hits  / total_h)
        near_history.append(nears / total_h)
        avg_reward_history.append(sum(m_rewards) / total_h)

        if len(m_rewards) > 1:
            manager.update_policy(
                m_states, m_wide, m_narrow, m_rewards, m_values, m_log_probs, m_entropies,
                step_count=episode, balls_snapshots=m_balls,
                narrow_only=True, ppo_epochs=ppo_epochs,
            )

        if use_per:
            for dx_s, dy_s, wg_s, ng_s, rew_s, lp_s in zip(m_dx, m_dy, m_wide, m_narrow, m_rewards, m_log_probs):
                replay_buffer.add(dx_s, dy_s, wg_s, ng_s, rew_s, lp_s.item())

            ep1 = episode + 1
            if (ep1 >= per_warmup and (ep1 - per_warmup) % per_replay_freq == 0
                    and len(replay_buffer) >= per_batch_size):
                beta    = per_beta_start + (1.0 - per_beta_start) * min(1.0, ep1 / max(1, episodes))
                n_steps = len(replay_buffer) // per_batch_size

                # lower lr for replay only
                for pg in manager.optimizer.param_groups:
                    pg['_saved_lr'] = pg['lr']
                    pg['lr']        = pg['lr'] * per_lr_factor

                per_loss_sum, per_ent_sum, per_steps_done = 0.0, 0.0, 0
                for _ in range(n_steps):
                    samples, weights = replay_buffer.sample(per_batch_size, beta)
                    new_lps, old_lps, advantages, entropies_r = [], [], [], []
                    for (dx_r, dy_r, wg_r, ng_r, rew_r, old_lp_r), w in zip(samples, weights):
                        nl = manager.narrow_head(torch.tensor([dx_r, dy_r], dtype=torch.float32, device=device))
                        nb = manager.get_neighborhood(wg_r)
                        vi = [i for i, c in enumerate(nb) if c in valid_set]
                        gi = nb.index(ng_r) if ng_r in nb else None
                        li = vi.index(gi) if gi is not None and gi in vi else None
                        if li is not None and vi:
                            vp = F.softmax(nl[torch.tensor(vi, dtype=torch.long, device=device)], dim=0)
                            new_lps.append(torch.log(vp[li] + 1e-8))
                            old_lps.append(float(old_lp_r))
                            advantages.append(float(w) * float(rew_r))
                            entropies_r.append(-(vp * torch.log(vp + 1e-8)).sum())
                    if new_lps:
                        new_lp_t = torch.stack(new_lps)
                        old_lp_t = torch.tensor(old_lps, dtype=torch.float32, device=device)
                        adv_t    = torch.tensor(advantages, dtype=torch.float32, device=device)
                        ent_t    = torch.stack(entropies_r).mean()
                        ratios   = torch.exp(new_lp_t - old_lp_t)
                        surr1    = ratios * adv_t
                        surr2    = ratios.clamp(1 - 0.2, 1 + 0.2) * adv_t
                        loss     = -torch.min(surr1, surr2).mean() - per_entropy_coef * ent_t
                        manager.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(manager.narrow_head.parameters(), max_norm=0.5)
                        manager.optimizer.step()
                        per_loss_sum += loss.item()
                        per_ent_sum  += ent_t.item()
                        per_steps_done += 1
                if per_steps_done > 0:
                    print(f"    [PER ep {ep1}] steps={per_steps_done} | "
                          f"loss={per_loss_sum/per_steps_done:.4f} | "
                          f"ent={per_ent_sum/per_steps_done:.3f}")

                # restore lr
                for pg in manager.optimizer.param_groups:
                    pg['lr'] = pg['_saved_lr']

        if (episode + 1) % 50 == 0:
            recent_hit  = hit_history[-50:]
            recent_near = near_history[-50:]
            avg_entropy = sum(m_entropies).item() / len(m_entropies) if m_entropies else 0.0
            print(f"  Ep {episode+1:>5}/{episodes} | "
                  f"Hit(dist=0): {sum(recent_hit)/len(recent_hit)*100:.1f}% | "
                  f"Near(dist<=1): {sum(recent_near)/len(recent_near)*100:.1f}% | "
                  f"Entropy: {avg_entropy:.3f}")

        if len(hit_history) >= 50 and all(h == 1.0 for h in hit_history[-50:]):
            print(f"  Early stopping at ep {episode+1}: 50 consecutive episodes at 100% hit rate.")
            break

    manager.entropy_coef = _original_entropy_coef

    final_hit = sum(hit_history[-100:]) / min(100, len(hit_history)) * 100
    print(f"\nManager Narrow Pre-training complete. Final hit rate (last 100 ep): {final_hit:.1f}%")
    plot_manager_narrow_pretrain_diagnostics(hit_history, near_history, avg_reward_history)
    return hit_history

#----------------------------------------------------------------------------#
#                     PHASE 3: INTEGRATION TRAINING                          #
#----------------------------------------------------------------------------#

def _run_phase3_training(config, pivotal_states, world_graph, policy, env,
                         agent_start, first_balls, session_path, grid_state,
                         phase3_animation=True):
    
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
        maze_size=config['maze_size'].value,
        neighborhood_size=config['neighborhood_size'],
        device=config['device'],
    )
    manager.initialize_from_goal_policy(policy)
    worker.initialize_from_goal_policy(policy)

    run_worker_pretrain(env, worker, grid_state, config, config['device'])
    worker.goal_norm_div = 1.0  # keep consistent with pretrain (raw cell deltas, no normalization)

    if config.get('worker_edge_refine', True):
        world_graph = refine_edges_with_worker(
            env, worker, world_graph, pivotal_states, config, config['device'])
        worker.world_graph    = world_graph
        worker.pivotal_states = set(tuple(p) for p in pivotal_states)

    if config.get('manager_wide_pretrain_episodes', 0) > 0:
        run_manager_wide_pretrain(env, manager, grid_state, config, config['device'])

    if config.get('manager_narrow_pretrain_episodes', 0) > 0:
        run_manager_narrow_pretrain(env, manager, grid_state, config, config['device'])

    # Reset optimizer for Phase 3 with lower LR (fresh momentum, avoids pretrain gradient bleed)
    phase3_lr = config.get('manager_phase3_lr', config['manager_lr'])
    manager.optimizer = torch.optim.Adam(manager.parameters(), lr=phase3_lr)
    print(f"Manager optimizer reset for Phase 3: lr={phase3_lr:.1e}")

    # Pre-training phases modify env state — restore correct Phase 3 configuration
    env.agent_start_pos = agent_start
    env.agent_start_dir = 0
    env.phase = 2
    env.fixed_ball_positions = first_balls   # manager pretrain cleared this; restore it
    env.randomgen = True  # pretrain sets this to False; restore so env.reset() calls ResetMultiGoals

    print("\nDiagnosing Worker behavior BEFORE training:")
    diagnose_worker_behavior_single_episode(env, manager, worker, world_graph)

    trainer = HierarchicalTrainer(
        manager, worker, env,
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],
        diagnostic_checkstart=config['diagnostic_checkstart'],
        goal_timeout=config.get('goal_timeout', 3),
        traversal_shaping_weight=config.get('traversal_shaping_weight', 2.0),
        instant_traversal=config.get('phase3_train_with_instant_traversals', False),
    )

    print("\nPHASE 3: Hierarchical Training")
    metrics = {
        'rewards': [], 'steps': [], 'manager_updates': [],
        'worker_updates': [], 'times': [],
    }

    debug_interval = max(1, config['phase3_episodes'] // 20)
    for episode in range(config['phase3_episodes']):
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
            print(f"\n--- Episode {episode+1}/{config['phase3_episodes']} | reward={stats['episode_reward']:.2f} | entropy={stats['manager_entropy']:.3f} | balls={stats['balls_collected']}/{trainer.env.total_balls} ---")

    print("\n" + "="*70)
    print("PHASE 3 COMPLETE")
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

    if phase3_animation:
        for _ep_i in range(5):
            _run_and_save_episode(
                manager, worker, config, grid_state, agent_start, first_balls,
                f'phase3_final_episode_{_ep_i + 1}.mp4', fps=15, max_steps=500,
                world_graph=world_graph, pivotal_states=pivotal_states,
            )

    return metrics

#----------------------------------------------------------------------------#
#                         STANDALONE ENTRY POINTS                            #
#----------------------------------------------------------------------------#

def run_phase3_standalone(
        checkpoint_path='phase1_checkpoint.pt',
        config_overrides=None,
        fixed_balls=True,
        phase3_animation=True):
    
    """Run Phase 3 training using a saved Phase 1 checkpoint.
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
    env.phase = 2

    # Reachability diagnostic: check from nearest pivotal state to agent_start
    nearest_pivotal = min(pivotal_states, key=lambda p: manhattan_distance(p, agent_start))
    reachable = world_graph.get_reachable_nodes(nearest_pivotal)
    print(f"Graph reachability from nearest pivotal {nearest_pivotal}: {len(reachable)}/{len(world_graph.nodes)} nodes reachable")
    unreachable = [n for n in pivotal_states if n not in reachable]
    print(f"  Unreachable: {unreachable[:10]}{'...' if len(unreachable) > 10 else ''}")

    if fixed_balls:
        first_balls = env.ResetMultiGoals(agent_start, goals=config.get('num_balls', 5))
        env.fixed_ball_positions = first_balls
        print(f"Fixed ball positions: {first_balls}")
    else:
        first_balls = None
        print("Ball positions: random each episode")

    session_path = checkpoint_path.replace('.pt', '_session.pt')
    metrics = _run_phase3_training(
        config, pivotal_states, world_graph, policy, env,
        agent_start, first_balls, session_path, grid_state, phase3_animation,
    )

    print(f"Best reward: {max(metrics['rewards']):.2f}")
    print(f"Final 10-ep avg: {sum(metrics['rewards'][-10:]) / 10:.2f}")
    return metrics

def run_worker_pretrain_standalone(
        use_checkpoint=True,
        checkpoint_path='phase1_checkpoint_MEDIUM.pt',
        config_overrides=None,
        save_path=None):
    """
    Train the Worker/GCP pre-training phase in isolation.

    use_checkpoint=True : load maze layout + existing GCP from a Phase 1 checkpoint.
    use_checkpoint=False: generate a fresh random maze (like Phase 1 env setup) and a
                          fresh GCP — no Phase 1 training loop, no VAE, no graph.

    No balls, no world graph, no Manager. Pure GCP navigation pre-training.

    Args:
        use_checkpoint: whether to load maze + GCP from checkpoint_path.
        checkpoint_path: Phase 1 .pt file (ignored when use_checkpoint=False).
        config_overrides: dict of keys to override in externalconfig.
        save_path: if given, saves fine-tuned GCP state dict here.

    Returns:
        policy (GCP) after pre-training.
    """
    config = dict(externalconfig)
    if config_overrides:
        config.update(config_overrides)
    resolve_device(config)
    device = config['device']

    if use_checkpoint:
        _, _, policy, _, _, grid_state = load_phase1_checkpoint(checkpoint_path)
        policy.to(device)
        env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL,
                              max_steps=config['max_steps_per_episode'])
        env.reset()
        restore_maze_from_grid_state(env, grid_state)
        env.reset()
        print(f"Loaded maze from checkpoint: {checkpoint_path}")
    else:
        env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL,
                              max_steps=config['max_steps_per_episode'])
        env.phase = 1
        env.randomgen = True
        env.reset()
        grid_state = env.getGridState()
        policy = GoalConditionedPolicy(lr=config['goal_policy_lr'], maze_size=config['maze_size'].value, device=device)
        print(f"Fresh maze ({config['maze_size'].name}), fresh GCP.")

    # No graph needed — run_worker_pretrain bypasses traversal entirely
    worker = HierarchicalWorker(
        world_graph=None,
        pivotal_states=[],
        lr=config['worker_lr'],
        goal_policy=policy,
        maze_size=config['maze_size'].value,
        neighborhood_size=config.get('neighborhood_size', 3),
        device=device,
    )
    worker.initialize_from_goal_policy(policy)

    run_worker_pretrain(env, worker, grid_state, config, device)

    if save_path:
        torch.save({'goal_policy': policy.state_dict()}, save_path)
        print(f"GCP weights saved to {save_path}")

    return policy

def run_manager_wide_narrow_pretrain_standalone(
        use_checkpoint=True,
        checkpoint_path='phase1_checkpoint_MEDIUM.pt',
        config_overrides=None,
        save_path=None):
    """Run Manager wide pretrain then narrow pretrain back to back."""
    config = dict(externalconfig)
    if config_overrides:
        config.update(config_overrides)
    resolve_device(config)
    device = config['device']

    if use_checkpoint:
        pivotal_states, _, _, _, _, grid_state = load_phase1_checkpoint(checkpoint_path)
        env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL,
                              max_steps=config['max_steps_per_episode'])
        env.reset()
        restore_maze_from_grid_state(env, grid_state)
        env.reset()
        print(f"Loaded maze + {len(pivotal_states)} pivotal states from: {checkpoint_path}")
    else:
        env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL,
                              max_steps=config['max_steps_per_episode'])
        env.phase = 1
        env.randomgen = True
        env.reset()
        grid_state = env.getGridState()
        valid_cells = [
            (x, y)
            for x in range(1, env.width - 1)
            for y in range(1, env.height - 1)
            if env._is_traversable(env.grid.get(x, y))
        ]
        n = config.get('num_fake_pivotal_states', 30)
        pivotal_states = random.sample(valid_cells, min(n, len(valid_cells)))
        print(f"Fresh maze ({config['maze_size'].name}), {len(pivotal_states)} random pivotal states.")

    manager = HierarchicalManager(
        pivotal_states,
        neighborhood_size=config['neighborhood_size'],
        lr=config['manager_lr'],
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],
        diagnostic_checkstart=config['diagnostic_checkstart'],
        action_verbose=False,
        device=device,
    )
    manager.initialize_from_goal_policy(GoalConditionedPolicy(
        lr=config['goal_policy_lr'], maze_size=config['maze_size'].value, device=device))

    run_manager_wide_pretrain(env, manager, grid_state, config, device)
    run_manager_narrow_pretrain(env, manager, grid_state, config, device)

    if save_path:
        torch.save({'manager': manager.state_dict()}, save_path)
        print(f"Manager weights saved to {save_path}")

    return manager

#----------------------------------------------------------------------------#
#                             FULL PIPELINE                                  #
#----------------------------------------------------------------------------#

def train_full_phase1_to_phase3(
        config=externalconfig,
        phase3_animation=True):
    
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
    
    policy = GoalConditionedPolicy(lr=config['goal_policy_lr'], maze_size=config['maze_size'].value, device=config['device'])
    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=config['vae_mu0'], grid_size=env.size)
    buffer = StatBuffer()
    
    print("\nPHASE 1: World Graph Discovery")
    start_time = time.time()
    
    pivotal_states, world_graph, stat_buffer = alternating_training_loop(
        env, policy, vae_system, buffer, max_iterations=config['phase1_iterations'],
        convergence_threshold=config.get('convergence_threshold', 0.01),
        explore_top_fraction=config.get('explore_top_fraction', 0.20),
        diversity_walk_number=config.get('diversity_walk_number', 10),
        walk_length=config.get('walk_length', 250),
        walk_bias=config.get('walk_bias', 0.65),
        walk_episodes=config.get('walk_episodes', 4),
        graph_walk_length=config.get('graph_walk_length', 20),
        graph_num_attempts=config.get('graph_num_attempts', 70),
        spread_alpha=config.get('pivotal_spread_alpha', 0.0),
        skip_gcp_refine=config.get('worker_edge_refine', True),
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

    if len(pivotal_states) < 2:
        print(f"\nERROR: Phase 1 produced only {len(pivotal_states)} pivotal state(s). "
              f"Phase 3 requires at least 2. Check VAE training — try increasing phase1_iterations or vae_mu0.")
        return

    checkpoint_path = f"phase1_checkpoint_{config['maze_size'].name}.pt"
    save_phase1_checkpoint(checkpoint_path, pivotal_states, world_graph, policy, vae_system, config, GRIDSTATE)

    session_path = checkpoint_path.replace('.pt', '_session.pt')
    metrics = _run_phase3_training(
        config, pivotal_states, world_graph, policy, env,
        env.agent_start_pos, None, session_path, GRIDSTATE, phase3_animation,
    )

    print(f"Phase 1 time: {phase1_time:.1f}s")
    print(f"Phase 3 time: {sum(metrics['times']):.1f}s")
    print(f"Best reward: {max(metrics['rewards']):.2f}")
    print(f"Final 10-ep avg: {sum(metrics['rewards'][-10:])/10:.2f}")
    print(f"Avg manager updates/ep: {sum(metrics['manager_updates'])/len(metrics['manager_updates']):.1f}")
    print(f"Avg worker updates/ep: {sum(metrics['worker_updates'])/len(metrics['worker_updates']):.1f}")

#----------------------------------------------------------------------------#
#                                  MAIN                                      #
#----------------------------------------------------------------------------#

def main():
    """
    test_phase1_with_diagnostics(config={
        'maze_size': externalconfig['maze_size'],
        'phase1_iterations': externalconfig['phase1_iterations'],
        'vae_mu0': externalconfig['vae_mu0'],
        'device': externalconfig['device'],
    })
    """
    #train_full_phase1_to_phase3()       # Phase 1 + Phase 3 together (saves checkpoint automatically)
    #run_worker_pretrain_standalone(use_checkpoint=True,  checkpoint_path='phase1_checkpoint_MEDIUM.pt', config_overrides=externalconfig, save_path='gcp_pretrained.pt')
    run_worker_pretrain_standalone(use_checkpoint=False, config_overrides=externalconfig)
    #run_manager_wide_narrow_pretrain_standalone(use_checkpoint=False, checkpoint_path='phase1_checkpoint_MEDIUM.pt', config_overrides=externalconfig, save_path='manager_pretrained.pt')
    #run_phase3_standalone('phase1_checkpoint_MEDIUM.pt', config_overrides=externalconfig, fixed_balls=True, phase3_animation=False)
    #render_phase3_episode_gif('phase1_checkpoint_MEDIUM.pt', filename='phase3_final_episode.mp4', fps=15, max_steps=500)

if __name__ == "__main__":
    main()
