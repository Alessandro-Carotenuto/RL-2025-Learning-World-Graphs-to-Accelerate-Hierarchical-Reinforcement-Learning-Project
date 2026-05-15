# EXTERNAL LIBRARY IMPORTS

import pygame
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math

from enum import Enum

import minigrid
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall, Ball
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

from collections import deque

from typing import Any, Iterable, SupportsFloat, TypeVar
from gymnasium.core import ActType, ObsType

import time

import heapq

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Uniform

from typing import List, Tuple, Dict, Optional

# PROJECT-SPECIFIC IMPORTS

from local_distributions.hardkuma import HardKumaraswamy,BetaDistribution
from wrappers.minigrid_wrapper import MinigridWrapper,EnvModes,EnvSizes
from wrappers.fast_wrapper import FastWrapper
from utils.graph_manager import GraphManager, GraphVisualizer
from utils.statistics_buffer import StatBuffer,TestBuffer
from utils.statistics_visualizer import Visualizer
from local_networks.vaesystem import PriorNetwork,InferenceNetwork,GenerationNetwork
from local_networks.vaesystem import StateEncoder, ActionEncoder, VAESystem
from local_networks.policy_networks import GoalConditionedPolicy
from utils.misc import manhattan_distance,sample_goal_position
from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker, HierarchicalTrainer
from utils.optimal_reward_computer import compute_optimal_reward_for_episode, compute_optimal_reward_bruteforce_small



import pickle
import imageio
def replay_and_save_video(env_config, episode_data, filename):
    """Replay episode and save as video."""
    env = MinigridWrapper(
        size=env_config['maze_size'],
        mode=EnvModes.MULTIGOAL,
        max_steps=env_config['max_steps_per_episode'],
        render_mode='rgb_array'
    )
    env.phase = 2
    
    # Restore grid state
    env.grid = Grid(env.size, env.size)
    for y, row in enumerate(episode_data['grid_state']):
        for x, cell in enumerate(row):
            if cell == '#':
                env.grid.set(x, y, Wall())
            elif cell == 'B':
                env.grid.set(x, y, Ball(COLOR_NAMES[0]))
    
    # Set agent
    env.agent_pos = episode_data['initial_agent_pos']
    env.agent_dir = episode_data['initial_agent_dir']
    env.active_balls = set(episode_data['ball_positions'])
    
    # Restore step_count if present, else set to 0 to avoid AttributeError
    if hasattr(env, 'step_count'):
        if 'step_count' in episode_data:
            env.step_count = episode_data['step_count']
        else:
            env.step_count = 0
    else:
        # If env does not have step_count, create it
        env.step_count = episode_data.get('step_count', 0)

    frames = []
    for action in episode_data['actions']:
        frames.append(env.render())
        env.step(action)

    imageio.mimsave(filename, frames, fps=10)
    print(f"Saved video: {filename}")

#---------------------------------------------------------------------------------------
# MAIN ALTERNATING TRAINING LOOP - UPDATED TO INCLUDE WORLD GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------------------

def print_grid_image(GRIDTEXT,name=' '):
    fig, ax = plt.subplots(figsize=(len(GRIDTEXT[0]), len(GRIDTEXT)))
    ax.set_xlim(0, len(GRIDTEXT[0]))
    ax.set_ylim(0, len(GRIDTEXT))
    ax.set_aspect('equal')
    ax.axis('off')
    
    for i, row in enumerate(GRIDTEXT):
        for j, char in enumerate(row):
            ax.text(j + 0.5, len(GRIDTEXT) - i - 0.5, char, 
                    ha='center', va='center', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('grid'+name+'.png', dpi=150, bbox_inches='tight')

def _walk_away_from_spawn(env, spawn: tuple, walk_length: int = 400, bias: float = 0.7) -> tuple:
    """
    Random walk biased toward moving away from spawn.
    At each step: if move_forward increases manhattan distance from spawn,
    take it with probability `bias`; otherwise pick a random action.
    Returns the position reached. No map knowledge required — only env.step().
    """
    obs = env.reset()
    current_pos = tuple(env.agent_pos)
    dir_delta = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    for _ in range(walk_length):
        dx, dy = dir_delta[env.agent_dir]
        fwd = (current_pos[0] + dx, current_pos[1] + dy)
        if (manhattan_distance(fwd, spawn) > manhattan_distance(current_pos, spawn)
                and random.random() < bias):
            action = 2  # move_forward (away from spawn)
        else:
            action = random.choice([0, 1, 2])
        try:
            obs, _, term, trunc, _ = env.step(action)
            current_pos = tuple(env.agent_pos)
            if term or trunc:
                break
        except Exception:
            break
    return current_pos


def alternating_training_loop(env, policy, vae_system, buffer, max_iterations: int = 8, fast_training=True,
                              explore_top_fraction: float = 0.20,
                              diversity_walk_number: int = 10,
                              walk_length: int = 250,
                              walk_bias: float = 0.65,
                              walk_episodes: int = 4,
                              graph_walk_length: int = 20,
                              graph_num_attempts: int = 70):
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
    'policy_success_rates': []
    }


    # NEW: Persistent KL weight across iterations
    persistent_kl_weight = 1.0

    # Ramping parameters (not used now)
    # total_epochs = max_iterations * 25  # Assuming 25 epochs per iteration
    # kl_ramp_rate = -0.5 / (total_epochs * 0.5) # Ramp from 0.5 to 1.0 over half the total epochs

    # Compute spawn position once (it's fixed throughout Phase 1)
    _obs = env.reset()
    spawn_pos = tuple(env.agent_pos)
    print(f"Spawn position: {spawn_pos}")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        print(f"Current KL weight: {persistent_kl_weight:.3f}")
        
        # Collect initial data if needed
        if buffer.episodes_in_buffer < 3:
            print("Not enough episodes for VAE training, collecting initial data...")
            for _ in range(5):
                obs = env.reset()
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
            pivotal_states = vae_system.train(
                buffer, 
                num_epochs=25, 
                batch_size=8,
                initial_kl_weight=persistent_kl_weight,
                annealing_rate=0.0  # No annealing within iteration
            )
        except Exception as e:
            print(f"VAE training failed: {e}")
            continue
        
        # do not update persistent_kl_weight here
        # persistent_kl_weight = max(0.5, persistent_kl_weight + kl_ramp_rate * 25)
        
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
            
            # Start curiosity only after first iteration
            if iteration == 0:
                curiosity_weight = 0.0  # Pure goal-seeking first
                use_curiosity = False
            else:
                curiosity_weight = max(0.15, 0.5 - (iteration * 0.05))  # Decay: 0.5→0.15
                use_curiosity = True
            
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

            if fast_training:
                threshold_reconstruction_loss=0.01
            else:
                threshold_reconstruction_loss=0.005

            if avg_change < threshold_reconstruction_loss:
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

# PLOT AND DIAGNOSTICS -------------------------------------------------------

def diagnose_graph_connectivity(world_graph, pivotal_states, env):
    """
    Diagnose graph connectivity issues from spawn position.
    """
    print("\n" + "="*70)
    print("GRAPH CONNECTIVITY DIAGNOSTICS")
    print("="*70)
    
    # Get spawn position
    env.phase = 2
    obs = env.reset()
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
                path, dist = world_graph.shortest_path(start, end)
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

def diagnose_worker_behavior_single_episode(env, manager, worker, world_graph, pivotal_states):
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
        wide_goal, narrow_goal, log_prob, value, entropy = manager.get_manager_action(current_pos)
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
            except:
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

def plot_training_diagnostics(trainer, config, save_path=None):
    """Plot training diagnostics: 6 panels covering task, manager, and worker signals."""

    def moving_average(data, window=20):
        if len(data) < window:
            return np.array([])
        return np.convolve(data, np.ones(window)/window, mode='valid')

    history = trainer.diagnostic_history
    num_episodes = len(history['episode_rewards'])
    episodes = list(range(1, num_episodes + 1))
    ma_start = 20  # moving average window

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hierarchical RL Training Diagnostics', fontsize=16, fontweight='bold')

    def plot_with_ma(ax, data, color, label, ylabel, title, ylim=None):
        ax.plot(episodes, data, color=color, linewidth=1.5, alpha=0.5, label=label)
        ma = moving_average(data)
        if len(ma) > 0:
            ax.plot(range(ma_start, ma_start + len(ma)), ma, 'k-', linewidth=2, label=f'MA({ma_start})')
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # 1. Episode Rewards vs Optimal
    ax = axes[0, 0]
    ax.plot(episodes, history['episode_rewards'], 'b-', linewidth=1.5, alpha=0.5, label='Agent')
    ma_r = moving_average(history['episode_rewards'])
    if len(ma_r) > 0:
        ax.plot(range(ma_start, ma_start + len(ma_r)), ma_r, 'b-', linewidth=2, label=f'MA({ma_start})')
    ax.set_title('Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. Balls Collected per Episode
    plot_with_ma(axes[0, 1],
                 history['balls_collected_per_episode'],
                 'green', 'Balls', 'Balls Collected',
                 'Balls Collected per Episode',
                 ylim=[0, 6])

    # 3. Manager Goal Diversity  (unique goals / total horizons — drops as manager converges)
    plot_with_ma(axes[0, 2],
                 history['manager_goal_diversity'],
                 'orange', 'Diversity', 'Unique Goals / Horizons',
                 'Manager Goal Diversity\n(↓ = converging on fewer goals)',
                 ylim=[0, 1])

    # 4. Manager Entropy  (absolute nats — drops as policy peaks)
    max_entropy = np.log(len(trainer.manager.pivotal_states))
    entropy_pct = [e / max_entropy * 100 for e in history['manager_entropy']]
    plot_with_ma(axes[1, 0],
                 entropy_pct,
                 'red', 'Entropy %', '% of Max Entropy',
                 'Manager Policy Entropy\n(↓ = more decisive)',
                 ylim=[80, 101])

    # 5. Avg Distance: Manager Goals → Nearest Ball  (drops as manager learns ball locations)
    dist_data = history['goal_distance_to_balls']
    if dist_data:
        dist_episodes = list(range(1, len(dist_data) + 1))
        ax5 = axes[1, 1]
        ax5.plot(dist_episodes, dist_data, 'purple', linewidth=1.5, alpha=0.5, label='Dist')
        ma_d = moving_average(dist_data)
        if len(ma_d) > 0:
            ax5.plot(range(ma_start, ma_start + len(ma_d)), ma_d, 'k-', linewidth=2, label=f'MA({ma_start})')
        ax5.set_title('Avg Distance: Manager Goals → Balls\n(↓ = manager targeting balls)')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Manhattan Distance')
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=8)

    # 6. Manager Value Estimate  (rises and stabilises as critic converges)
    plot_with_ma(axes[1, 2],
                 history['manager_value_mean'],
                 'cyan', 'Value', 'Average Value',
                 'Manager Value Estimate\n(stabilises when critic converges)')

    plt.tight_layout()
    if save_path is None:
        save_path = f"diagnostics_size{config['maze_size'].name}_h{config['manager_horizon']}_n{config['neighborhood_size']}_ep{config['phase2_episodes']}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nDiagnostic plots saved to {save_path}")
    plt.close()


def save_separate_graph_visualization(world_graph, pivotal_states, config, grid_state=None):
    """
    Save a standalone visualization of the world graph, rendering the
    actual feasible paths for each edge.
    
    Args:
        world_graph (GraphManager): The fully constructed world graph instance.
        pivotal_states (list): The list of discovered pivotal states.
        config (dict): Configuration dictionary to get parameters like vae_mu0.
    """
    # --- Step 1: Basic validation ---
    if not pivotal_states:
        print("No pivotal states to visualize. Skipping graph saving.")
        return
        
    if world_graph is None or not world_graph.nodes:
        print("World graph is empty or not provided. Skipping graph saving.")
        return

    print("Generating and saving world graph visualization...")
    
    try:
        # --- Step 2: Instantiate the UPDATED visualizer ---
        # This visualizer now knows how to read the {'weight': w, 'path': p}
        # structure from the world_graph.edges.
        viz = GraphVisualizer(world_graph, figsize=(12, 12)) # Slightly larger for clarity
        
        # --- Step 3: Generate the visualization ---
        # The .visualize() method will automatically plot the detailed coordinate
        # paths instead of simple straight lines. No change is needed in this call.
        fig, ax = viz.visualize(
            show_weights=True,
            show_labels=True,
            node_size=250,
            edge_width=1.5,
            title=f'World Graph (mu0={config["vae_mu0"]}) - Feasible Paths',
            grid_state=grid_state
        )
        
        # --- Step 4: Save the figure ---
        filename = f'world_graph_mu{config["vae_mu0"]:.1f}.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight') # Higher DPI for better quality
        plt.close(fig)
        
        print(f"Successfully saved graph visualization to '{filename}'")

    except Exception as e:
        print(f"An error occurred while saving the graph visualization: {e}")
        # Ensure the plot is closed even if an error occurs
        if 'fig' in locals():
            plt.close(fig)

def create_phase1_gif(all_pivotal_states_history, grid_state, filename='phase1_evolution.gif', fps=2):
    """One frame per Phase 1 iteration — same style as the final graph visualization."""
    if not all_pivotal_states_history:
        print("No Phase 1 history to animate.")
        return

    frames = []
    total = len(all_pivotal_states_history)

    for iteration, pivotal_states in enumerate(all_pivotal_states_history):
        temp_graph = GraphManager()
        for ps in pivotal_states:
            temp_graph.add_node(ps)

        viz = GraphVisualizer(temp_graph, figsize=(10, 10))
        fig, ax = viz.visualize(
            show_weights=False,
            show_labels=True,
            node_size=250,
            edge_width=1.5,
            title=f'Phase 1 — Iteration {iteration + 1}/{total}  |  {len(pivotal_states)} pivotal states',
            grid_state=grid_state,
        )

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(filename, frames, fps=fps)
    print(f"Phase 1 evolution GIF saved to '{filename}' ({len(frames)} frames)")


def render_phase2_episode_gif(checkpoint_path, filename='phase2_final_episode.mp4', fps=15, max_steps=500):
    """
    Standalone: load checkpoint + session file, run one greedy episode, save as MP4.
    Call this after training from anywhere — no training objects needed.
    Requires: checkpoint .pt  +  checkpoint _session.pt (saved automatically at end of Phase 2).
    """
    pivotal_states, world_graph, policy, vae_system, config, grid_state = load_phase1_checkpoint(checkpoint_path)

    # Fall back to sensible defaults for older Phase 1 checkpoints
    config.setdefault('max_steps_per_episode', 2000)
    config.setdefault('neighborhood_size', math.ceil(config['maze_size'].value / 4))
    config.setdefault('manager_horizon', config['max_steps_per_episode'] // 120)
    config.setdefault('manager_lr', 5e-4)
    config.setdefault('worker_lr', 1e-4)
    config.setdefault('diagnostic_interval', 10000)
    config.setdefault('diagnostic_checkstart', False)

    session_path = checkpoint_path.replace('.pt', '_session.pt')
    session = torch.load(session_path, map_location='cpu', weights_only=False)
    agent_start = session['agent_start']
    ball_positions = session['ball_positions']

    # Apply Phase 2 fine-tuned GCP weights if saved in session
    if 'goal_policy_state_dict' in session:
        policy.load_state_dict(session['goal_policy_state_dict'])

    manager = HierarchicalManager(
        pivotal_states,
        neighborhood_size=config['neighborhood_size'],
        lr=config['manager_lr'],
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],
        diagnostic_checkstart=config['diagnostic_checkstart'],
        device='cpu',
    )
    manager.load_state_dict(session['manager_state_dict'])
    manager.eval()

    worker = HierarchicalWorker(
        world_graph,
        pivotal_states,
        lr=config['worker_lr'],
        goal_policy=policy,
        device='cpu',
    )
    worker.load_state_dict(session['worker_state_dict'])
    worker.eval()

    _run_and_save_episode(manager, worker, config, grid_state, agent_start, ball_positions, filename, fps, max_steps)


def render_phase2_episode_gif_from_objects(manager, worker, config, grid_state, agent_start_pos,
                                           ball_positions=None, filename='phase2_final_episode.mp4',
                                           fps=15, max_steps=500):
    """
    Run one greedy episode with trained manager/worker and save as MP4.
    Called at end of training when all objects are in memory.
    """
    _run_and_save_episode(manager, worker, config, grid_state, agent_start_pos, ball_positions, filename, fps, max_steps)


def _run_and_save_episode(manager, worker, config, grid_state, agent_start_pos,
                          ball_positions, filename, fps, max_steps):
    env = MinigridWrapper(
        size=config['maze_size'],
        mode=EnvModes.MULTIGOAL,
        max_steps=config['max_steps_per_episode'],
        render_mode='rgb_array',
    )
    env.reset()
    restore_maze_from_grid_state(env, grid_state)
    env.agent_start_pos = agent_start_pos
    env.agent_pos = agent_start_pos
    env.placeable_grid[agent_start_pos[0]][agent_start_pos[1]] = False
    env.firstgen = False
    env.phase = 2
    if ball_positions is not None:
        env.fixed_ball_positions = ball_positions

    env.reset()
    state = tuple(env.agent_pos)
    manager.reset_manager_state()
    worker.reset_worker_state()

    frames = [env.render()]
    done = False
    step = 0
    horizon_step = 0
    wide_goal = manager.pivotal_states[0]
    narrow_goal = manager.pivotal_states[0]

    with torch.no_grad():
        while not done and step < max_steps:
            if horizon_step == 0:
                wide_goal, narrow_goal, _, _, _ = manager.get_manager_action(state, step_count=999999)
                if manager.hidden_state is not None:
                    manager.hidden_state = tuple(h.detach() for h in manager.hidden_state)
            action, _, _ = worker.get_action(state, wide_goal, narrow_goal, agent_dir=env.agent_dir)
            try:
                obs, _, terminated, truncated, _ = env.step(action)
            except (AssertionError, IndexError):
                terminated, truncated = False, False
            state = tuple(env.agent_pos)
            frames.append(env.render())
            done = terminated or truncated
            step += 1
            horizon_step = (horizon_step + 1) % config['manager_horizon']

    writer = imageio.get_writer(filename, fps=fps, format='ffmpeg')
    for frame in frames:
        writer.append_data(np.array(frame, dtype=np.uint8))
    writer.close()
    print(f"Phase 2 video saved to '{filename}' ({len(frames)} frames, {len(frames)/fps:.1f}s)")


def test_phase1_with_diagnostics(config=None):
    """
    Test Phase 1 using alternating_training_loop with diagnostic tracking. 
    """
    default_config = {
        'maze_size': EnvSizes.MEDIUM,
        'phase1_iterations': 15,
        'vae_mu0': 10.0,
        'goal_policy_lr': 5e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
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
        graph_num_attempts=config.get('graph_num_attempts', 70)
    )
    
    # Extract metrics from VAE training history
    metrics = {
        'vae_losses': [h['total_loss'] for h in vae_system.training_history],
        'vae_reconstruction': [h['reconstruction_loss'] for h in vae_system.training_history],
        'vae_kl': [h['kl_divergence'] for h in vae_system.training_history],
        'vae_l0': [h['expected_l0'] for h in vae_system.training_history],
        'num_pivotal_states': loop_metrics['num_pivotal_states_per_iteration'],  # CHANGED
        'policy_episodes': buffer.episodes_in_buffer,
        'policy_success_rate': loop_metrics['policy_success_rates']  # CHANGED
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


    GRIDSTATE=env.getGridState()

    # Generate plots (around line 800)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # Changed from (2, 2)

    # Plot 1: VAE Total Loss
    if metrics['vae_losses']:
        axes[0, 0].plot(metrics['vae_losses'], 'b-', linewidth=2)
        axes[0, 0].set_title('VAE Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Reconstruction Loss
    if metrics['vae_reconstruction']:
        axes[0, 1].plot(metrics['vae_reconstruction'], 'r-', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: KL Divergence
    if metrics['vae_kl']:
        axes[0, 2].plot(metrics['vae_kl'], 'g-', linewidth=2)
        axes[0, 2].set_title('KL Divergence (Should Stay > 0.01)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].axhline(y=0.01, color='orange', linestyle='--', label='Floor')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Expected L0
    if metrics['vae_l0']:
        axes[1, 0].plot(metrics['vae_l0'], 'purple', linewidth=2, label='Actual')
        axes[1, 0].axhline(y=config['vae_mu0'], color='orange', linestyle='--', label='Target')
        axes[1, 0].set_title('Expected L0 (Sparsity)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Pivotal States Discovered (NEW)
    if metrics['num_pivotal_states']:
        axes[1, 1].plot(metrics['num_pivotal_states'], 'cyan', linewidth=2, marker='o')
        axes[1, 1].set_title('Pivotal States Discovered')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Policy Success Rate (NEW)
    if metrics['policy_success_rate']:
        axes[1, 2].plot(metrics['policy_success_rate'], 'magenta', linewidth=2, marker='s')
        axes[1, 2].set_title('Goal Policy Success Rate')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('Success Rate')
        axes[1, 2].set_ylim([0, 1])
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'phase1_diagnostics_mu{config["vae_mu0"]:.1f}.png', dpi=150)
    print(f"\nSaved diagnostics to phase1_diagnostics_mu{config['vae_mu0']:.1f}.png")
    plt.close()

    save_separate_graph_visualization(world_graph, pivotal_states, config, grid_state=GRIDSTATE)
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

#STILL TO USE ------------------------------------------------------------
def analyze_phase1_metrics(results):
    """Extract key metrics and diagnose Phase 1 issues."""
    
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
    """Compare multiple Phase 1 runs with different parameters."""
    
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
#                        PHASE 1 CHECKPOINT SAVE / LOAD                      #
#----------------------------------------------------------------------------#

def save_phase1_checkpoint(path, pivotal_states, world_graph, policy, vae_system, config, grid_state):
    """Save all Phase 1 outputs to a single file."""
    checkpoint = {
        'pivotal_states': pivotal_states,
        'world_graph': world_graph,
        'policy_state_dict': policy.state_dict(),
        'vae_state_dict': vae_system.state_dict(),
        'vae_kwargs': {
            'state_dim': 16,
            'action_vocab_size': 7,
            'mu0': config['vae_mu0'],
            'grid_size': int(config['maze_size'].value) + 4,
        },
        'config': config,
        'grid_state': grid_state,
    }
    torch.save(checkpoint, path)
    print(f"Phase 1 checkpoint saved to '{path}'")


def load_phase1_checkpoint(path, device=None):
    """Load Phase 1 checkpoint. Returns (pivotal_states, world_graph, policy, vae_system, config, grid_state)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    config = checkpoint['config']
    config.setdefault('max_steps_per_episode', 2000)
    config.setdefault('neighborhood_size', math.ceil(config['maze_size'].value / 4))
    config.setdefault('manager_horizon', config['max_steps_per_episode'] // 120)
    config.setdefault('manager_lr', 5e-4)
    config.setdefault('worker_lr', 1e-4)
    config.setdefault('diagnostic_interval', 10000)
    config.setdefault('diagnostic_checkstart', False)
    if device is not None:
        config['device'] = device

    vae_kw = checkpoint['vae_kwargs']
    vae_system = VAESystem(
        state_dim=vae_kw['state_dim'],
        action_vocab_size=vae_kw['action_vocab_size'],
        mu0=vae_kw['mu0'],
        grid_size=vae_kw['grid_size'],
    )
    vae_system.load_state_dict(checkpoint['vae_state_dict'])
    vae_system.to(config['device'])

    policy = GoalConditionedPolicy(lr=5e-3, device=config['device'])
    policy.load_state_dict(checkpoint['policy_state_dict'])

    print(f"Phase 1 checkpoint loaded from '{path}'")
    print(f"  Pivotal states: {len(checkpoint['pivotal_states'])}")
    print(f"  Graph edges:    {len(checkpoint['world_graph'].edges)}")

    return (
        checkpoint['pivotal_states'],
        checkpoint['world_graph'],
        policy,
        vae_system,
        config,
        checkpoint['grid_state'],
    )


def restore_maze_from_grid_state(env, grid_state):
    """Overwrite env.grid with the Phase 1 maze walls from the ASCII grid_state."""
    h = len(grid_state)
    w = len(grid_state[0]) if h > 0 else 0
    for y in range(h):
        for x in range(w):
            char = grid_state[y][x]
            if char == '#':
                env.grid.set(x, y, Wall())
                env.placeable_grid[x][y] = False
            else:
                env.grid.set(x, y, None)
                env.placeable_grid[x][y] = True
    # Agent start position is never placeable
    env.placeable_grid[env.agent_start_pos[0]][env.agent_start_pos[1]] = False


def run_phase2_standalone(checkpoint_path='phase1_checkpoint.pt', config_overrides=None, fixed_balls=True,
                          phase2_animation=True):
    """Run Phase 2 training using a saved Phase 1 checkpoint.
    fixed_balls=True : same ball positions every episode (manager can learn spatial strategy)
    fixed_balls=False: random ball positions every episode
    """
    pivotal_states, world_graph, policy, vae_system, config, grid_state = load_phase1_checkpoint(checkpoint_path)

    if config_overrides:
        config.update(config_overrides)

        # If device is overridden, move loaded models to the new device too.
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested in config_overrides but not available. Falling back to CPU.")
            config['device'] = 'cpu'

        vae_system.to(config['device'])
        policy.to(config['device'])

    env = MinigridWrapper(
        size=config['maze_size'],
        mode=EnvModes.MULTIGOAL,
        max_steps=config['max_steps_per_episode'],
    )
    env.reset()  # triggers _gen_grid → initializes grid + placeable_grid
    restore_maze_from_grid_state(env, grid_state)

    # Find a valid start position in the Phase 1 maze: not a wall, ≥6 reachable cells
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

    env.firstgen = False  # prevent re-generation on next reset
    env.phase = 2         # next reset → ResetMultiGoals → balls placed

    # Graph reachability diagnostic
    reachable = world_graph.get_reachable_nodes(agent_start)
    print(f"Graph reachability from {agent_start}: {len(reachable)}/{len(world_graph.nodes)} nodes reachable")
    unreachable = [n for n in pivotal_states if n not in reachable]
    print(f"  Unreachable from start: {unreachable[:10]}{'...' if len(unreachable) > 10 else ''}")

    if fixed_balls:
        first_balls = env.ResetMultiGoals(agent_start, goals=5)
        env.fixed_ball_positions = first_balls
        print(f"Fixed ball positions: {first_balls}")
    else:
        first_balls = None
        print("Ball positions: random each episode")

    session_path = checkpoint_path.replace('.pt', '_session.pt')
    torch.save({'agent_start': agent_start, 'ball_positions': first_balls}, session_path)
    print(f"Session saved to '{session_path}'")

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
    diagnose_worker_behavior_single_episode(env, manager, worker, world_graph, pivotal_states)

    trainer = HierarchicalTrainer(
        manager, worker, env,
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],
        diagnostic_checkstart=config['diagnostic_checkstart'],
        goal_timeout=config.get('goal_timeout', 3),
    )

    print("\nPHASE 2: Hierarchical Training (standalone)")
    metrics = {'rewards': [], 'steps': [], 'manager_updates': [], 'worker_updates': [], 'times': [], 'optimal_rewards': []}

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
        metrics['optimal_rewards'].append(stats['optimal_reward'])
        if episode % debug_interval == 0 and episode > 0:
            print(f"\n--- Episode {episode+1}/{config['phase2_episodes']} | reward={stats['episode_reward']:.2f} | entropy={stats['manager_entropy']:.3f} | balls={stats['balls_collected']}/{trainer.env.total_balls} ---")

    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    plot_training_diagnostics(trainer, config)

    # Update session with trained weights (GCP fine-tuned in Phase 2)
    session_path = checkpoint_path.replace('.pt', '_session.pt')
    torch.save({
        'agent_start': agent_start,
        'ball_positions': first_balls,
        'manager_state_dict': manager.state_dict(),
        'worker_state_dict': worker.state_dict(),
        'goal_policy_state_dict': policy.state_dict(),
    }, session_path)
    print(f"Session updated with trained weights: '{session_path}'")

    if phase2_animation:
        render_phase2_episode_gif_from_objects(
            manager, worker, config, grid_state,
            agent_start_pos=agent_start,
            ball_positions=first_balls,
        )

    print(f"Best reward: {max(metrics['rewards']):.2f}")
    print(f"Final 10-ep avg: {sum(metrics['rewards'][-10:]) / 10:.2f}")
    return metrics


# ACTUAL TRAINING CODE ----------------------------------------------------
steps=2000

externalconfig = {
        'maze_size': EnvSizes.MEDIUM,
        'phase1_iterations': 2,
        'phase2_episodes': 10,
        'max_steps_per_episode': steps,
        'manager_horizon': steps//250,
        'neighborhood_size': math.ceil(24/4),
        'manager_lr': 5e-4,
        'worker_lr': 1e-4,
        'vae_mu0': 9.0,
        'diagnostic_interval': 10000,
        'diagnostic_checkstart': False,
        'full_breakdown_every': 10,
        'goal_timeout': 3,              # max horizons before forcing a new Manager goal
        'explore_top_fraction': 0.20,   # Phase 1: top % of pivotal states (by dist from spawn) used for trajectory collection
        'diversity_walk_number': 30,    # Phase 1: biased random walks per iteration
        'walk_length': 400,             # Phase 1: steps per diversity walk
        'walk_bias': 0.70,              # Phase 1: probability of stepping away from spawn
        'walk_episodes': 10,             # Phase 1: episodes collected per walk destination
        'graph_walk_length': 50,         # Phase 1: max steps per random walk for edge discovery
        'graph_num_attempts': 150,       # Phase 1: random walk attempts per pivotal state
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

fast_training_toggle=True

def train_full_phase1_phase2(config=externalconfig, fast_training=fast_training_toggle, recordflag=False,
                             phase1_animation=True, phase2_animation=True):
    """Complete training with comprehensive diagnostics."""
    # Hyperparameters setted up in externalconfig

    # Validate device availability
    if config['device'] == 'cuda':
        if not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available. Falling back to CPU.")
            config['device'] = 'cpu'
        else:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    print("="*70)
    print("FULL TRAINING with Diagnostics")
    print("="*70)
    
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Phase 1
    env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL, max_steps=config['max_steps_per_episode'],phase_one_eps=config['phase1_iterations']*10000)
    env.phase = 1
    env.randomgen = True
    
    policy = GoalConditionedPolicy(lr=5e-3,device=config['device']) # Hardcoded learning rate for goal policy
    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=config['vae_mu0'], grid_size=env.size)
    buffer = StatBuffer()
    
    print("\nPHASE 1: World Graph Discovery")
    start_time = time.time()
    
    pivotal_states, world_graph, stat_buffer, all_pivotal_states = alternating_training_loop(
        env, policy, vae_system, buffer, max_iterations=config['phase1_iterations'],
        fast_training=fast_training,
        explore_top_fraction=config.get('explore_top_fraction', 0.20),
        diversity_walk_number=config.get('diversity_walk_number', 10),
        walk_length=config.get('walk_length', 250),
        walk_bias=config.get('walk_bias', 0.65),
        walk_episodes=config.get('walk_episodes', 4),
        graph_walk_length=config.get('graph_walk_length', 20),
        graph_num_attempts=config.get('graph_num_attempts', 70)
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
    save_separate_graph_visualization(world_graph, pivotal_states, config, grid_state=GRIDSTATE)

    if phase1_animation:
        create_phase1_gif(all_pivotal_states, GRIDSTATE)

    checkpoint_path = f"phase1_checkpoint_{config['maze_size'].name}.pt"
    save_phase1_checkpoint(checkpoint_path, pivotal_states, world_graph, policy, vae_system, config, GRIDSTATE)

    if len(pivotal_states) < 2:
        print(f"\nERROR: Phase 1 produced only {len(pivotal_states)} pivotal state(s). "
              f"Phase 2 requires at least 2. Check VAE training — try increasing phase1_iterations or vae_mu0.")
        return

    # After phase 1, before phase 2 setup:
    if recordflag:
        grid_state = env.getGridState()
        recording_data = {
            'bad_episode': None,
            'good_episode': None,
            'grid_state': grid_state,
            'config': config
        }

    # Phase 2: PASS THE LEARNING RATES AND DIAGNOSTIC PARAMS!
    manager = HierarchicalManager(
        pivotal_states, 
        neighborhood_size=config['neighborhood_size'],
        lr=config['manager_lr'],
        horizon=config['manager_horizon'],
        diagnostic_interval=config['diagnostic_interval'],  # NEW
        diagnostic_checkstart=config['diagnostic_checkstart'],  # NEW
        device=config['device']
    )
    worker = HierarchicalWorker(
        world_graph,
        pivotal_states,
        lr=config['worker_lr'],
        goal_policy=policy,
        device=config['device']
    )
    manager.initialize_from_goal_policy(policy)
    worker.initialize_from_goal_policy(policy)

    print("\nDiagnosing Worker behavior BEFORE training:")
    diagnose_worker_behavior_single_episode(
        env, manager, worker, world_graph, pivotal_states
    )
    
    

    env.phase = 2
    trainer = HierarchicalTrainer(
    manager, worker, env,
    horizon=config['manager_horizon'],
    diagnostic_interval=config['diagnostic_interval'],
    diagnostic_checkstart=config['diagnostic_checkstart'],
    goal_timeout=config.get('goal_timeout', 3))
    
    print("\nPHASE 2: Hierarchical Training")
    
    # Tracking
    metrics = {
        'rewards': [],
        'steps': [],
        'manager_updates': [],
        'worker_updates': [],
        'traversals': [],
        'times': [],
        'optimal_rewards': []
    }
    
    debug_interval = max(1, config['phase2_episodes'] // 20)
    for episode in range(config['phase2_episodes']):
        ep_start = time.time()
        stats = trainer.train_episode(
            max_steps=config['max_steps_per_episode'],
            full_breakdown_every=debug_interval,
            recording_data=recording_data if recordflag else None
        )

        metrics['rewards'].append(stats['episode_reward'])
        metrics['steps'].append(stats['episode_steps'])
        metrics['manager_updates'].append(stats['manager_updates'])
        metrics['worker_updates'].append(stats['worker_updates'])
        metrics['times'].append(time.time() - ep_start)
        metrics['optimal_rewards'].append(stats['optimal_reward'])
        if episode % debug_interval == 0 and episode > 0:
            print(f"\n--- Episode {episode+1}/{config['phase2_episodes']} | reward={stats['episode_reward']:.2f} | entropy={stats['manager_entropy']:.3f} | balls={stats['balls_collected']}/{trainer.env.total_balls} ---")
    
    # AFTER all episodes complete - NOW plot the diagnostics
    print("\n" + "="*70)
    print("TRAINING COMPLETE - Generating diagnostic plots...")
    print("="*70)
    
    plot_training_diagnostics(trainer,config)  # ← HERE, after the loop

    # Results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Phase 1 time: {phase1_time:.1f}s")
    print(f"Phase 2 time: {sum(metrics['times']):.1f}s")
    print(f"Best reward: {max(metrics['rewards']):.2f}")
    print(f"Final 10-ep avg: {sum(metrics['rewards'][-10:])/10:.2f}")
    print(f"Avg manager updates/ep: {sum(metrics['manager_updates'])/len(metrics['manager_updates']):.1f}")
    print(f"Avg worker updates/ep: {sum(metrics['worker_updates'])/len(metrics['worker_updates']):.1f}")
    
    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(metrics['rewards'], 'b-', label='Agent')
    axes[0, 0].plot(metrics['optimal_rewards'], 'r--', label='Optimal')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episodes')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(metrics['steps'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(metrics['manager_updates'], label='Manager')
    axes[1, 0].plot(metrics['worker_updates'], label='Worker')
    axes[1, 0].set_title('Updates per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(metrics['times'])
    axes[1, 1].set_title('Time per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Seconds')
    axes[1, 1].grid(True, alpha=0.3)
    

    # Plots - BOTH with config in filename
    simple_plot_path = (f"training_simple_"
                       f"size{config['maze_size'].name}_"
                       f"h{config['manager_horizon']}_"
                       f"ep{config['phase2_episodes']}.png")
    
    plt.tight_layout()
    plt.savefig(simple_plot_path)
    print("\nPlots saved to training_diagnostics.png")

        # After plots, before return
    if recordflag and recording_data['bad_episode'] is not None:
        replay_and_save_video(config, recording_data['bad_episode'], 'bad_episode.mp4')
    if recordflag and recording_data['good_episode'] is not None:
        replay_and_save_video(config, recording_data['good_episode'], 'good_episode.mp4')

    # Save trained weights to session (GCP fine-tuned in Phase 2)
    checkpoint_path = f"phase1_checkpoint_{config['maze_size'].name}.pt"
    session_path = checkpoint_path.replace('.pt', '_session.pt')
    torch.save({
        'agent_start': env.agent_start_pos,
        'ball_positions': None,
        'manager_state_dict': manager.state_dict(),
        'worker_state_dict': worker.state_dict(),
        'goal_policy_state_dict': policy.state_dict(),
    }, session_path)
    print(f"Session updated with trained weights: '{session_path}'")

    if phase2_animation:
        render_phase2_episode_gif_from_objects(
            manager, worker, config, GRIDSTATE,
            agent_start_pos=env.agent_start_pos,
        )


def run_phase1_comparison():
    """
    Run Phase 1 training 3 times on the SAME map with different mu0 values.
    """
    mu0_values = [3.0, 6.0, 9.0]
    maze_size = EnvSizes.MEDIUM
    iterations = 50
    
    # Detect available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create ONE environment that will be reused
    base_env = MinigridWrapper(
        size=maze_size, 
        mode=EnvModes.MULTIGOAL,
        phase_one_eps=iterations * 10000
    )
    base_env.phase = 1
    base_env.randomgen = True  # Generate once
    
    # Generate the map once
    print("Generating base map...")
    base_env.reset()  # This creates self.h and self.w
    base_grid_state = base_env.getGridState()
    print_grid_image(base_grid_state, name='base_map')
    
    # NOW disable random generation for subsequent runs
    base_env.randomgen = False
    base_env.firstgen = False
    
    results = {}
    
    for mu0 in mu0_values:
        print(f"\n{'='*70}")
        print(f"Running Phase 1 with mu0={mu0}")
        print(f"{'='*70}")
        
        # Create fresh networks
        policy = GoalConditionedPolicy(lr=5e-3, device=device)
        vae_system = VAESystem(
            state_dim=16,
            action_vocab_size=7,
            mu0=mu0,
            grid_size=base_env.size,
            device='cpu'
        )
        buffer = StatBuffer()
        
        # Use the base_env directly (already has the generated map)
        base_env.phase = 1
        
        # Run training
        pivotal_states, world_graph, metrics, _ = alternating_training_loop(
            base_env, policy, vae_system, buffer,
            max_iterations=iterations,
            fast_training=True
        )
        
        # Save results
        results[f'mu0_{mu0}'] = {
            'pivotal_states': pivotal_states,
            'world_graph': world_graph,
            'metrics': metrics,
            'vae_system': vae_system,
            'buffer': buffer
        }
        
        # Generate plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        axes[0, 0].plot([h['total_loss'] for h in vae_system.training_history], 'b-')
        axes[0, 0].set_title(f'VAE Loss (mu0={mu0})')
        
        axes[0, 1].plot([h['reconstruction_loss'] for h in vae_system.training_history], 'r-')
        axes[0, 1].set_title('Reconstruction Loss')
        
        axes[0, 2].plot([h['kl_divergence'] for h in vae_system.training_history], 'g-')
        axes[0, 2].set_title('KL Divergence')
        
        axes[1, 0].plot([h['expected_l0'] for h in vae_system.training_history], 'purple')
        axes[1, 0].axhline(y=mu0, color='orange', linestyle='--', label='Target')
        axes[1, 0].set_title('Expected L0')
        axes[1, 0].legend()
        
        axes[1, 1].plot(metrics['num_pivotal_states_per_iteration'], 'cyan', marker='o')
        axes[1, 1].set_title('Pivotal States Discovered')
        
        axes[1, 2].plot(metrics['policy_success_rates'], 'magenta', marker='s')
        axes[1, 2].set_title('Policy Success Rate')
        axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'phase1_comparison_mu{mu0:.1f}.png', dpi=150)
        plt.close()
        
        save_separate_graph_visualization(world_graph, pivotal_states, {'vae_mu0': mu0})
        
        print(f"\n✓ Completed mu0={mu0}")
        print(f"  Pivotal states: {len(pivotal_states)}")
        print(f"  Graph edges: {len(world_graph.edges)}")
    
    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'mu0':<8} {'Pivotal':<10} {'Edges':<8} {'Final Loss':<12}")
    print("-" * 70)
    
    for mu0 in mu0_values:
        key = f'mu0_{mu0}'
        res = results[key]
        final_loss = res['vae_system'].training_history[-1]['total_loss']
        print(f"{mu0:<8.1f} {len(res['pivotal_states']):<10} {len(res['world_graph'].edges):<8} {final_loss:<12.4f}")
    
    return results

def run_phase1_size_comparison():
    """
    Run Phase 1 training on different environment sizes with mu0=9.0.
    """
    sizes = [EnvSizes.SMALL, EnvSizes.MEDIUM]
    mu0 = 9.0
    iterations = 50
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = {}
    
    for size in sizes:
        print(f"\n{'='*70}")
        print(f"Running Phase 1 with size={size.name} (grid={size.value}x{size.value})")
        print(f"{'='*70}")
        
        # Create environment for this size
        env = MinigridWrapper(
            size=size,
            mode=EnvModes.MULTIGOAL,
            phase_one_eps=iterations * 10000
        )
        env.phase = 1
        env.randomgen = True
        
        # Generate and save map
        env.reset()
        grid_state = env.getGridState()
        print_grid_image(grid_state, name=f'map_{size.name}')
        
        # Create networks
        policy = GoalConditionedPolicy(lr=5e-3, device=device)
        vae_system = VAESystem(
            state_dim=16,
            action_vocab_size=7,
            mu0=mu0,
            grid_size=env.size,
            device=device
        )
        buffer = StatBuffer()
        
        # Train
        pivotal_states, world_graph, metrics, _ = alternating_training_loop(
            env, policy, vae_system, buffer,
            max_iterations=iterations,
            fast_training=True
        )
        
        results[size.name] = {
            'pivotal_states': pivotal_states,
            'world_graph': world_graph,
            'metrics': metrics,
            'vae_system': vae_system,
            'size': size.value
        }
        
        # Generate plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        axes[0, 0].plot([h['total_loss'] for h in vae_system.training_history], 'b-')
        axes[0, 0].set_title(f'VAE Loss ({size.name})')
        
        axes[0, 1].plot([h['reconstruction_loss'] for h in vae_system.training_history], 'r-')
        axes[0, 1].set_title('Reconstruction Loss')
        
        axes[0, 2].plot([h['kl_divergence'] for h in vae_system.training_history], 'g-')
        axes[0, 2].set_title('KL Divergence')
        
        axes[1, 0].plot([h['expected_l0'] for h in vae_system.training_history], 'purple')
        axes[1, 0].axhline(y=mu0, color='orange', linestyle='--', label='Target')
        axes[1, 0].set_title('Expected L0')
        axes[1, 0].legend()
        
        axes[1, 1].plot(metrics['num_pivotal_states_per_iteration'], 'cyan', marker='o')
        axes[1, 1].set_title('Pivotal States Discovered')
        
        axes[1, 2].plot(metrics['policy_success_rates'], 'magenta', marker='s')
        axes[1, 2].set_title('Policy Success Rate')
        axes[1, 2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'phase1_size_{size.name}.png', dpi=150)
        plt.close()
        
        save_separate_graph_visualization(world_graph, pivotal_states, {'vae_mu0': mu0})
        
        print(f"\n✓ Completed {size.name}")
        print(f"  Grid size: {size.value}x{size.value}")
        print(f"  Pivotal states: {len(pivotal_states)}")
        print(f"  Graph edges: {len(world_graph.edges)}")
    
    # Comparison summary
    print(f"\n{'='*70}")
    print("SIZE COMPARISON SUMMARY (mu0=9.0)")
    print(f"{'='*70}")
    print(f"{'Size':<10} {'Grid':<8} {'Pivotal':<10} {'Edges':<8} {'Final Loss':<12}")
    print("-" * 70)
    
    for size_name, res in results.items():
        final_loss = res['vae_system'].training_history[-1]['total_loss']
        print(f"{size_name:<10} {res['size']}x{res['size']:<6} {len(res['pivotal_states']):<10} {len(res['world_graph'].edges):<8} {final_loss:<12.4f}")
    
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
    train_full_phase1_phase2(recordflag=False)       # Phase 1 + Phase 2 together (saves checkpoint automatically)
    #run_phase2_standalone('phase1_checkpoint_MEDIUM.pt', config_overrides=externalconfig, fixed_balls=True, phase2_animation=True)  # fixed_balls=False for random
    #render_phase2_episode_gif('phase1_checkpoint_MEDIUM.pt', filename='phase2_final_episode.mp4', fps=15, max_steps=500)
    # run_phase1_comparison()
    # run_phase1_size_comparison()


if __name__ == "__main__":
    main()
