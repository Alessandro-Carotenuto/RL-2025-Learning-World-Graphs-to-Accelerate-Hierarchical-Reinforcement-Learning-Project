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

def alternating_training_loop(env, policy, vae_system, buffer, max_iterations: int = 8, fast_training=True):
    """
    Main alternating training loop with persistent KL annealing.
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
                    env, start_pos, num_episodes=6, vae_system=vae_system
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
        print(f"Second Half: Collecting trajectories from top {min(10, len(pivotal_states))} pivotal states...")
        
        # Use fewer pivotal states early, more as training progresses
        num_pivotal_to_use = min(5 + iteration, len(pivotal_states))


        episodes_collected = 0
        success_count = 0
        for i, start_state in enumerate(pivotal_states[:10]):  
            print(f"  Collecting from pivotal state {i+1}: {start_state}")
            
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
                    max_episode_length=50,  # Increased from 25
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
        
        # Check for convergence
        if len(reconstruction_losses) >= 3:
            recent_losses = reconstruction_losses[-3:]
            loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
            avg_change = sum(loss_changes) / len(loss_changes)
            
            print(f"Average loss change over last 3 iterations: {avg_change:.5f}")
            
            if fast_training:
                #threshold_reconstruction_loss=0.05
                threshold_reconstruction_loss=0.01
            else:
                threshold_reconstruction_loss=0.005

            if avg_change < threshold_reconstruction_loss:
                print("Reconstruction loss has plateaued - training converged!")
                break
        
        # Optional diversity collection
        if iteration % 2 == 0:
            print("Phase 4: Adding diversity with random exploration...")
            obs = env.reset()
            random_start = tuple(env.agent_pos)
            try:
                random_episodes = policy.collect_episodes_from_position(
                    env, random_start, num_episodes=6, vae_system=vae_system
                )
                if random_episodes:
                    buffer.add_episodes(random_episodes)
            except Exception as e:
                print(f"Random exploration failed: {e}")
    

    # Construct world graph
    world_graph = policy.complete_world_graph_discovery(env, pivotal_states)
    
    # Final summary
    print(f"\nAlternating Training Complete!")
    print(f"Total iterations: {len(reconstruction_losses)}")
    print(f"Final episodes in buffer: {buffer.episodes_in_buffer}")
    print(f"Final pivotal states ({len(pivotal_states)}): {pivotal_states}")
    
    # CHANGE RETURN - Add metrics
    return pivotal_states, world_graph, metrics

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
        wide_goal, narrow_goal, log_prob, value = manager.get_manager_action(current_pos)
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
    """Plot comprehensive training diagnostics with moving averages."""
    import matplotlib.pyplot as plt

    
    def moving_average(data, window=20):
        """Compute moving average with specified window size."""
        if len(data) < window:
            return np.array([])  # Return empty array, not original data
        return np.convolve(data, np.ones(window)/window, mode='valid')
        
    history = trainer.diagnostic_history
    num_episodes = len(history['episode_rewards'])
    episodes = range(1, num_episodes + 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Hierarchical RL Training Diagnostics', fontsize=16, fontweight='bold')
    
    # 1. Episode rewards
    axes[0, 0].plot(episodes, history['episode_rewards'], 'b-', alpha=0.7, label='Agent')
    ma_rewards = moving_average(history['episode_rewards'])
    if len(ma_rewards) > 0:
        axes[0, 0].plot(range(20, 20 + len(ma_rewards)), ma_rewards, 'k-', linewidth=2, label='MA(20)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Manager goal diversity
    axes[0, 1].plot(episodes, history['manager_goal_diversity'], 'g-', linewidth=2, alpha=0.7, label='Diversity')
    ma_diversity = moving_average(history['manager_goal_diversity'])
    if len(ma_diversity) > 0:
        axes[0, 1].plot(range(20, 20 + len(ma_diversity)), ma_diversity, 'k-', linewidth=2, label='MA(20)')
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Random (100%)')
    axes[0, 1].axhline(y=1/len(trainer.manager.pivotal_states), color='orange', 
                       linestyle='--', alpha=0.5, label='Collapsed')
    axes[0, 1].set_title('Manager Goal Diversity')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Fraction of Unique Goals')
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Manager e
    axes[0, 2].plot(episodes, history['manager_entropy'], 'purple', linewidth=2, alpha=0.7, label='Entropy')
    ma_entropy = moving_average(history['manager_entropy'])
    if len(ma_entropy) > 0:
        axes[0, 2].plot(range(20, 20 + len(ma_entropy)), ma_entropy, 'k-', linewidth=2, label='MA(20)')
    max_entropy = np.log(len(trainer.manager.pivotal_states))
    axes[0, 2].axhline(y=max_entropy, color='r', linestyle='--', alpha=0.5, label='Max entropy')
    axes[0, 2].set_title('Manager Policy Entropy')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Entropy (nats)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # 4. Worker goal achievement
    axes[1, 0].plot(episodes, [x*100 for x in history['worker_goal_achievement']], 'orange', linewidth=2, alpha=0.7, label='Achievement')
    ma_achievement = moving_average([x*100 for x in history['worker_goal_achievement']])
    if len(ma_achievement) > 0:
        axes[1, 0].plot(range(20, 20 + len(ma_achievement)), ma_achievement, 'k-', linewidth=2, label='MA(20)')
    axes[1, 0].set_title('Worker Goal Achievement Rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('% Horizons Goal Reached')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 5. Balls collected
    axes[1, 1].plot(episodes, history['balls_collected_per_episode'], 'red', linewidth=2, alpha=0.7, label='Balls')
    ma_balls = moving_average(history['balls_collected_per_episode'])
    if len(ma_balls) > 0:
        axes[1, 1].plot(range(20, 20 + len(ma_balls)), ma_balls, 'k-', linewidth=2, label='MA(20)')
    axes[1, 1].axhline(y=trainer.env.total_balls, color='g', linestyle='--', 
                       alpha=0.5, label='All balls')
    axes[1, 1].set_title('Balls Collected per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Number of Balls')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 6. Manager rewards
    axes[1, 2].plot(episodes, history['manager_rewards_mean'], 'b-', linewidth=2, alpha=0.7, label='Mean')
    ma_manager_rew = moving_average(history['manager_rewards_mean'])
    if len(ma_manager_rew) > 0:
        axes[1, 2].plot(range(20, 20 + len(ma_manager_rew)), ma_manager_rew, 'k-', linewidth=2, label='MA(20)')
    axes[1, 2].fill_between(episodes, 
                            np.array(history['manager_rewards_mean']) - np.array(history['manager_rewards_std']),
                            np.array(history['manager_rewards_mean']) + np.array(history['manager_rewards_std']),
                            alpha=0.3)
    axes[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 2].set_title('Manager Reward per Horizon')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Reward')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # 7. Distance to balls
    if len(history['goal_distance_to_balls']) > 0:
        axes[2, 0].plot(episodes[:len(history['goal_distance_to_balls'])], 
                        history['goal_distance_to_balls'], 'brown', linewidth=2, alpha=0.7, label='Distance')
        ma_distance = moving_average(history['goal_distance_to_balls'])
        if len(ma_distance) > 0:
            axes[2, 0].plot(range(20, 20 + len(ma_distance)), ma_distance, 'k-', linewidth=2, label='MA(20)')
        axes[2, 0].set_title('Avg Distance: Manager Goals → Balls')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Manhattan Distance')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].legend()
    
    # 8. Manager values
    axes[2, 1].plot(episodes, history['manager_value_mean'], 'cyan', linewidth=2, alpha=0.7, label='Value')
    ma_manager_val = moving_average(history['manager_value_mean'])
    if len(ma_manager_val) > 0:
        axes[2, 1].plot(range(20, 20 + len(ma_manager_val)), ma_manager_val, 'k-', linewidth=2, label='MA(20)')
    axes[2, 1].set_title('Manager Value Estimates')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Average Value')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()
    
    # 9. Worker values
    axes[2, 2].plot(episodes, history['worker_value_mean'], 'magenta', linewidth=2, alpha=0.7, label='Value')
    ma_worker_val = moving_average(history['worker_value_mean'])
    if len(ma_worker_val) > 0:
        axes[2, 2].plot(range(20, 20 + len(ma_worker_val)), ma_worker_val, 'k-', linewidth=2, label='MA(20)')
    axes[2, 2].set_title('Worker Value Estimates')
    axes[2, 2].set_xlabel('Episode')
    axes[2, 2].set_ylabel('Average Value')
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].legend()
    
    plt.tight_layout()
    if save_path is None:
        # Build filename from config
        save_path = f"diagnostics_size{config['maze_size'].name}_h{config['manager_horizon']}_n{config['neighborhood_size']}_ep{config['phase2_episodes']}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nDiagnostic plots saved to {save_path}")
    plt.close()

def save_separate_graph_visualization(world_graph, pivotal_states, config):
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
            title=f'World Graph (mu0={config["vae_mu0"]}) - Feasible Paths'
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

def test_phase1_with_diagnostics(config=None):
    """
    Test Phase 1 using alternating_training_loop with diagnostic tracking. 
    """
    default_config = {
        'maze_size': EnvSizes.SMALL,
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
    pivotal_states, world_graph, loop_metrics = alternating_training_loop(
        env, policy, vae_system, buffer, 
        max_iterations=config['phase1_iterations']
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
    print_grid_image(GRIDSTATE,name=str(config['vae_mu0']))

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
    
    save_separate_graph_visualization(world_graph, pivotal_states, config)
    
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
# ACTUAL TRAINING CODE ----------------------------------------------------
steps=2000

externalconfig = {
        'maze_size': EnvSizes.MEDIUM,
        'phase1_iterations': 50,
        'phase2_episodes': 100,
        'max_steps_per_episode': steps,
        'manager_horizon': steps//120,
        'neighborhood_size': math.ceil(24/4),
        'manager_lr': 1e-4,
        'worker_lr': 1e-4,
        'vae_mu0': 9.0,
        'diagnostic_interval': 1000,  # NEW: Print diagnostics every K steps
        'diagnostic_checkstart': True,  # NEW: Print every step for first 15 steps
        'full_breakdown_every': 10,  # NEW: Full breakdown every N episodes
        'device': 'cuda'
    }

fast_training_toggle=True

def train_full_phase1_phase2(config=externalconfig, fast_training=fast_training_toggle, recordflag=False):
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
    
    pivotal_states, world_graph, stat_buffer = alternating_training_loop(
        env, policy, vae_system, buffer, max_iterations=config['phase1_iterations'],
        fast_training=fast_training
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
    print_grid_image(GRIDSTATE,name=str(config['vae_mu0']))
    save_separate_graph_visualization(world_graph, pivotal_states, config)

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
    diagnostic_interval=config['diagnostic_interval'],  # NEW
    diagnostic_checkstart=config['diagnostic_checkstart'])
    
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
    
    for episode in range(config['phase2_episodes']):
        ep_start = time.time()
        if episode%10 == 0 and episode > 0:
            print(f"\n--- Episode {episode+1}/{config['phase2_episodes']} ---")
        stats = trainer.train_episode(
        max_steps=config['max_steps_per_episode'],
        full_breakdown_every=config['full_breakdown_every'],
        recording_data=recording_data if recordflag else None
    )
        
        metrics['rewards'].append(stats['episode_reward'])
        metrics['steps'].append(stats['episode_steps'])
        metrics['manager_updates'].append(stats['manager_updates'])
        metrics['worker_updates'].append(stats['worker_updates'])
        metrics['times'].append(time.time() - ep_start)
        metrics['optimal_rewards'].append(stats['optimal_reward'])
        
        
        #OLD METRICS, NOW IN HIERARCHICAL SISTEM
        # if (episode + 1) % 10 == 0:
        #     recent = metrics['rewards'][-10:]
        #     print(f"Ep {episode+1}: avg_reward={sum(recent)/10:.2f}, "
        #           f"last={stats['episode_reward']:.2f}, "
        #           f"time={metrics['times'][-1]:.1f}s")
    
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
        pivotal_states, world_graph, metrics = alternating_training_loop(
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
        pivotal_states, world_graph, metrics = alternating_training_loop(
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
    # pygame.init()
    # env = MinigridWrapper(render_mode="human",size=EnvSizes.EXTRALARGE, mode=EnvModes.MULTIGOAL, phase_one_eps=10)
    # # #env = FastWrapper("MiniGrid-KeyCorridorS3R2-v0",1000,"human")
    # # print(os.path.dirname(minigrid.__file__))

    # # enable manual control for testing
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()

    train_full_phase1_phase2(recordflag=True)
    #run_phase1_comparison()
    #run_phase1_size_comparison()

    # config = {'vae_mu0': 9.0, 'phase1_iterations': 25, 'maze_size': EnvSizes.MEDIUM}
    # test_phase1_with_diagnostics(config)

    # Compare maze sizes
    # Compare different mu0 values
    # runs = {}
    # for currsize in [EnvSizes.MEDIUM]:
        #   config = {'vae_mu0': 9.0, 'phase1_iterations': 25, 'maze_size': EnvSizes.MEDIUM}
        #   runs[f'size={currsize}'] = test_phase1_with_diagnostics(config)
    
    # compare_phase1_runs(runs)

    # runs = {}
    # for size in [EnvSizes.SMALL, EnvSizes.MEDIUM]:
    #     config = {'maze_size': size}
    #     runs[f'{size.name}'] = test_phase1_with_diagnostics(config)

    # compare_phase1_runs(runs)

if __name__ == "__main__":
    main()
