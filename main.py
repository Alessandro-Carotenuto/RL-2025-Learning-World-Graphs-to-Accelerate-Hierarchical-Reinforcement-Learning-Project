# STANDARD IMPORTS

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

# LOCAL IMPORTS

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
from local_networks.hierarchical_system import hierarchical_system_tests, test_phase2_with_real_environment
from utils.optimal_reward_computer import compute_optimal_reward_for_episode, compute_optimal_reward_bruteforce_small

#---------------------------------------------------------------------------------------
# MAIN ALTERNATING TRAINING LOOP - UPDATED TO INCLUDE WORLD GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------------------

def alternating_training_loop(env, policy, vae_system, buffer, max_iterations: int = 8):
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
    persistent_kl_weight = 0.0
    total_epochs = max_iterations * 25  # Assuming 25 epochs per iteration
    kl_ramp_rate = 1.0 / (total_epochs * 0.5)  # Ramp over first 50% of total
    
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
                    env, start_pos, num_episodes=1, vae_system=vae_system
                )
                if episodes:
                    buffer.add_episodes(episodes)
            continue
        
        # Train VAE with persistent KL weight
        print(f"Phase 1: Training VAE on {buffer.episodes_in_buffer} episodes...")
        try:
            pivotal_states = vae_system.train(
                buffer, 
                num_epochs=25, 
                batch_size=3,
                initial_kl_weight=persistent_kl_weight  # NEW
            )
        except Exception as e:
            print(f"VAE training failed: {e}")
            continue
        
        # Update KL weight for next iteration
        persistent_kl_weight = min(1.0, persistent_kl_weight + kl_ramp_rate * 25)
        
        # Track metrics
        if vae_system.training_history:
            current_loss = vae_system.training_history[-1]['reconstruction_loss']
            reconstruction_losses.append(current_loss)
            print(f"Current reconstruction loss: {current_loss:.4f}")
        
        metrics['num_pivotal_states_per_iteration'].append(len(pivotal_states))

        all_pivotal_states.append(pivotal_states.copy())
        print(f"Discovered {len(pivotal_states)} pivotal states: {pivotal_states[:3]}...")
        
        # Phase 2: Collect trajectories from pivotal states
        print(f"Phase 2: Collecting trajectories from top {min(10, len(pivotal_states))} pivotal states...")
        
        episodes_collected = 0
        success_count = 0  # ADD THIS LINE
        for i, start_state in enumerate(pivotal_states[:10]):  # Increased from 5
            print(f"  Collecting from pivotal state {i+1}: {start_state}")
            
            curiosity_weight = max(0.2, 0.8 - (iteration * 0.1))
            
            try:
                episodes = policy.collect_episodes_from_position(
                    env, start_state,
                    num_episodes=2,
                    max_episode_length=40,  # Increased from 25
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
            
        # ADD THIS BLOCK - Track success rate
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
            
            if avg_change < 0.005:
                print("Reconstruction loss has plateaued - training converged!")
                break
        
        # Optional diversity collection
        if iteration % 2 == 0:
            print("Phase 4: Adding diversity with random exploration...")
            obs = env.reset()
            random_start = tuple(env.agent_pos)
            try:
                random_episodes = policy.collect_episodes_from_position(
                    env, random_start, num_episodes=1, vae_system=vae_system
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

#-----------------------------------------------------------------------------
# TEST FUNCTIONS FOR EACH COMPONENT, BEFORE SPLITTING IN FILES
#-----------------------------------------------------------------------------

def test_vae_system():
    """Test the VAE system with sample data."""
    print("Testing VAE System:")
    print("=" * 40)
    
    # Create sample stat buffer data
    sample_trajectories = [
        [((1, 1), 0), ((2, 1), 1), ((2, 2), 2), ((3, 2), 3)],  # Short trajectory
        [((5, 5), 1), ((5, 6), 0), ((4, 6), 2), ((4, 7), 1), ((3, 7), 3)],  # Medium trajectory
        [((10, 10), 2), ((11, 10), 0), ((11, 9), 3)],  # Another short trajectory
    ]
    
    # Mock StatBuffer for testing
    class MockStatBuffer:
        def __init__(self, trajectories):
            self.episodes_in_buffer = len(trajectories)
            self.trajectories = trajectories
        
        def get_STATE_ACT_traj_stream_byep(self):
            return self.trajectories
    
    mock_buffer = MockStatBuffer(sample_trajectories)
    
    # Create VAE system
    vae_system = VAESystem(
        state_dim=16,  # Smaller for testing
        action_vocab_size=7,
        hidden_dim=32,
        mu0=2.0,  # Expect 2 pivotal states per trajectory
        grid_size=24,
        device=config['device']
    )
    
    print(f"Created VAE system with {sum(p.numel() for p in vae_system.parameters())} parameters")
    
    # Test encoding
    print("\nTesting trajectory encoding...")
    states, actions, seq_lengths = vae_system.encode_trajectories(sample_trajectories)
    print(f"Encoded states shape: {states.shape}")
    print(f"Encoded actions shape: {actions.shape}")
    print(f"Sequence lengths: {seq_lengths}")
    
    # Test single training step
    print("\nTesting single training step...")
    loss_dict = vae_system.train_step(sample_trajectories)
    print("Loss components:")
    for key, value in loss_dict.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Test pivotal state extraction
    print("\nTesting pivotal state extraction...")
    pivotal_states = vae_system.extract_pivotal_states(sample_trajectories)
    print(f"Discovered {len(pivotal_states)} pivotal states: {pivotal_states}")
    
    print("\n" + "=" * 40)
    print("VAE System test completed!")

def test_goal_conditioned_policy():
    """Test the a implementation."""
    print("Testing Goal-Conditioned Policy:")
    print("=" * 40)
    
    # Create policy
    policy = GoalConditionedPolicy()
    print(f"Created policy with {sum(p.numel() for p in policy.parameters())} parameters")
    
    # Test forward pass
    print("\nTesting forward pass...")
    state = torch.tensor([5.0, 5.0])  # Agent at (5, 5)
    goal = torch.tensor([10.0, 8.0])  # Goal at (10, 8)
    
    action_logits, value = policy.forward(state, goal)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Sample action logits: {action_logits}")
    print(f"Sample value: {value.item():.3f}")
    
    # Test action sampling
    print("\nTesting action sampling...")
    action, log_prob, value = policy.get_action((5, 5), (10, 8))
    print(f"Sampled action: {action}")
    print(f"Log probability: {log_prob.item():.3f}")
    print(f"Value estimate: {value.item():.3f}")
    
    # Test utility functions
    print("\nTesting utility functions...")
    dist = manhattan_distance((5, 5), (10, 8))
    print(f"Manhattan distance from (5,5) to (10,8): {dist}")
    
    print("\n" + "=" * 40)
    print("Goal-Conditioned Policy test completed!")

def test_goal_policy_with_maze():
    """Test Goal-Conditioned Policy with actual MinigridWrapper environment."""
    print("Testing Goal-Conditioned Policy with Maze Environment:")
    print("=" * 60)
    
    # Create maze environment - use a simpler setup for goal training
    env = MinigridWrapper(
        size=EnvSizes.SMALL,  # Start with small maze (10x10)
        mode=EnvModes.MULTIGOAL  # Keep this but we'll manage it differently
    )
    
    # Set to phase 1 to avoid complex item placement
    env.phase = 1
    
    # Disable random generation to get consistent clean mazes
    env.randomgen = True
    
    # Create goal-conditioned policy
    policy = GoalConditionedPolicy(lr=1e-3)
    
    # Create buffer to store trajectories for future VAE training
    buffer = StatBuffer()
    
    print(f"Environment size: {env.size}x{env.size}")
    print(f"Agent starts at: {env.agent_start_pos}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    
    # Training loop
    num_training_episodes = 20
    success_count = 0
    
    for episode in range(num_training_episodes):
        # Reset environment
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Start position: {start_pos}")
        
        # Check if we can find reachable positions
        try:
            reachable_positions = env.BFS_all_reachable(start_pos)
            print(f"  Reachable positions: {len(reachable_positions)}")
            
            if len(reachable_positions) < 2:
                print("  Skipping - not enough reachable positions")
                continue
                
        except Exception as e:
            print(f"  Error finding reachable positions: {e}")
            continue
        
        # Train one episode with goal policy
        try:
            states, actions, rewards, goal_reached = policy.train_goal_policy_episode(
                env, start_pos, max_episode_length=30
            )
            
            print(f"  Episode length: {len(actions)}")
            print(f"  Total reward: {sum(rewards):.3f}")
            print(f"  Goal reached: {goal_reached}")
            
            if goal_reached:
                success_count += 1
            
            # Store trajectory in buffer for future VAE training
            for i, ((state, goal), action, reward) in enumerate(zip(states, actions, rewards)):
                next_state = states[i+1][0] if i+1 < len(states) else state
                done = (i == len(states) - 1)
                buffer.add(state, action, reward, next_state, done)
            
            buffer.end_episode()
            
        except Exception as e:
            print(f"  Error during episode: {e}")
            continue
    
    print(f"\nTraining Summary:")
    print(f"Episodes completed: {num_training_episodes}")
    print(f"Goals reached: {success_count}/{num_training_episodes}")
    print(f"Success rate: {success_count/num_training_episodes*100:.1f}%")
    print(f"Trajectories collected: {buffer.episodes_in_buffer}")
    print(f"Total steps: {buffer.total_steps_in_buffer}")
    
    # Test policy after training
    print(f"\nTesting trained policy:")
    test_policy_navigation(env, policy)
    
    return policy, buffer

def test_policy_navigation(env, policy, num_tests=3):
    """Test the trained policy on navigation tasks."""
    
    for test in range(num_tests):
        print(f"  Test {test + 1}:")
        
        # Reset and get start position
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        try:
            # Sample a goal
            goal_pos = sample_goal_position(env, start_pos, max_distance=5)
            print(f"    Navigate from {start_pos} to {goal_pos}")
            
            # Get action from trained policy
            action, log_prob, value = policy.get_action(start_pos, goal_pos)
            
            print(f"    Policy suggests action: {action}")
            print(f"    Action confidence: {torch.exp(log_prob).item():.3f}")
            print(f"    Value estimate: {value.item():.3f}")
            
            # Take the action and see result
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = tuple(env.agent_pos)
            
            # Check if action brought us closer to goal
            old_distance = manhattan_distance(start_pos, goal_pos)
            new_distance = manhattan_distance(new_pos, goal_pos)
            
            print(f"    New position: {new_pos}")
            print(f"    Distance change: {old_distance} → {new_distance} ({'better' if new_distance < old_distance else 'worse'})")
            
        except Exception as e:
            print(f"    Test failed: {e}")

def test_integration_compatibility():

    """Test that goal policy data is compatible with VAE training."""
    print("\nTesting VAE Integration Compatibility:")
    print("=" * 40)
    
    # Create simple test data
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    policy = GoalConditionedPolicy()
    buffer = StatBuffer()
    
    # Collect a few trajectories
    for ep in range(3):
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        try:
            states, actions, rewards, goal_reached = policy.train_goal_policy_episode(
                env, start_pos, max_episode_length=10
            )
            
            # Add to buffer
            for i, ((state, goal), action, reward) in enumerate(zip(states, actions, rewards)):
                next_state = states[i+1][0] if i+1 < len(states) else state
                done = (i == len(states) - 1)
                buffer.add(state, action, reward, next_state, done)
            
            buffer.end_episode()
            
        except Exception as e:
            print(f"Episode {ep} failed: {e}")
            continue
    
    print(f"Collected {buffer.episodes_in_buffer} episodes")
    
    # Test VAE compatibility
    if buffer.episodes_in_buffer > 0:
        try:
            # Get trajectories in VAE format
            trajectories = buffer.get_STATE_ACT_traj_stream_byep()
            print(f"Trajectory format: {len(trajectories)} episodes")
            
            if len(trajectories) > 0:
                print(f"Sample trajectory length: {len(trajectories[0])}")
                print(f"Sample state-action: {trajectories[0][0] if len(trajectories[0]) > 0 else 'None'}")
                
                # Test with VAE system (if available)
                try:
                    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=3.0)
                    states, actions, seq_lengths = vae_system.encode_trajectories(trajectories[:1])
                    print(f"VAE encoding successful:")
                    print(f"  States shape: {states.shape}")
                    print(f"  Actions shape: {actions.shape}")
                    print(f"  Sequence lengths: {seq_lengths}")
                    
                except Exception as e:
                    print(f"VAE encoding test failed: {e}")
                    
        except Exception as e:
            print(f"Trajectory extraction failed: {e}")
    
    print("Integration compatibility test completed!")
    """Test that goal policy data is compatible with VAE training."""
    print("\nTesting VAE Integration Compatibility:")
    print("=" * 40)
    
    # Create simple test data
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    policy = GoalConditionedPolicy()
    buffer = StatBuffer()
    
    # Collect a few trajectories
    for ep in range(3):
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        try:
            states, actions, rewards, goal_reached = policy.train_goal_policy_episode(
                env, start_pos, max_episode_length=10
            )
            
            # Add to buffer
            for i, ((state, goal), action, reward) in enumerate(zip(states, actions, rewards)):
                next_state = states[i+1][0] if i+1 < len(states) else state
                done = (i == len(states) - 1)
                buffer.add(state, action, reward, next_state, done)
            
            buffer.end_episode()
            
        except Exception as e:
            print(f"Episode {ep} failed: {e}")
            continue
    
    print(f"Collected {buffer.episodes_in_buffer} episodes")
    
    # Test VAE compatibility
    if buffer.episodes_in_buffer > 0:
        try:
            # Get trajectories in VAE format
            trajectories = buffer.get_STATE_ACT_traj_stream_byep()
            print(f"Trajectory format: {len(trajectories)} episodes")
            
            if len(trajectories) > 0:
                print(f"Sample trajectory length: {len(trajectories[0])}")
                print(f"Sample state-action: {trajectories[0][0] if len(trajectories[0]) > 0 else 'None'}")
                
                # Test with VAE system (if available)
                try:
                    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=3.0)
                    states, actions, seq_lengths = vae_system.encode_trajectories(trajectories[:1])
                    print(f"VAE encoding successful:")
                    print(f"  States shape: {states.shape}")
                    print(f"  Actions shape: {actions.shape}")
                    print(f"  Sequence lengths: {seq_lengths}")
                    
                except Exception as e:
                    print(f"VAE encoding test failed: {e}")
                    
        except Exception as e:
            print(f"Trajectory extraction failed: {e}")
    
    print("Integration compatibility test completed!")
    
def test_goal_policy_with_curiosity():
    """Test Goal-Conditioned Policy with VAE curiosity integration."""
    print("Testing Goal-Conditioned Policy with Curiosity Integration:")
    print("=" * 60)
    
    # Create maze environment
    env = MinigridWrapper(
        size=EnvSizes.SMALL,
        mode=EnvModes.MULTIGOAL
    )
    env.phase = 1
    env.randomgen = True
    
    # Create goal-conditioned policy
    policy = GoalConditionedPolicy(lr=1e-3)
    
    # Create VAE system for curiosity rewards
    vae_system = VAESystem(
        state_dim=16,
        action_vocab_size=7, 
        mu0=3.0,
        grid_size=env.size
    )
    
    # Create buffer to store trajectories
    buffer = StatBuffer()
    
    print(f"Environment size: {env.size}x{env.size}")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print(f"VAE parameters: {sum(p.numel() for p in vae_system.parameters())}")
    
    # Training loop with curiosity
    num_training_episodes = 25
    success_count = 0
    
    for episode in range(num_training_episodes):
        # Reset environment
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Start position: {start_pos}")
        
        # Check reachable positions
        try:
            reachable_positions = env.BFS_all_reachable(start_pos)
            print(f"  Reachable positions: {len(reachable_positions)}")
            
            if len(reachable_positions) < 2:
                print("  Skipping - not enough reachable positions")
                continue
                
        except Exception as e:
            print(f"  Error finding reachable positions: {e}")
            continue
        
        # Train one episode with curiosity-driven goal policy
        try:
            # Adjust curiosity weight over time (start high, decay)
            curiosity_weight = max(0.5, 2.0 - (episode / 15))  # Decay from 2.0 to 0.5
            
            states, actions, rewards, goal_reached = policy.train_goal_policy_episode(
                env, start_pos, 
                max_episode_length=40,
                vae_system=vae_system,
                curiosity_weight=curiosity_weight
            )
            
            print(f"  Episode length: {len(actions)}")
            print(f"  Total reward: {sum(rewards):.3f}")
            print(f"  Goal reached: {goal_reached}")
            print(f"  Curiosity weight: {curiosity_weight:.2f}")
            
            if goal_reached:
                success_count += 1
            
            # Store trajectory in buffer for VAE training
            for i, ((state, goal), action, reward) in enumerate(zip(states, actions, rewards)):
                next_state = states[i+1][0] if i+1 < len(states) else state
                done = (i == len(states) - 1)
                buffer.add(state, action, reward, next_state, done)
            
            buffer.end_episode()
            
        except Exception as e:
            print(f"  Error during episode: {e}")
            continue
    
    print(f"\nCuriosity Training Summary:")
    print(f"Episodes completed: {num_training_episodes}")
    print(f"Goals reached: {success_count}/{num_training_episodes}")
    print(f"Success rate: {success_count/num_training_episodes*100:.1f}%")
    print(f"Trajectories collected: {buffer.episodes_in_buffer}")
    print(f"Total steps: {buffer.total_steps_in_buffer}")
    
    # Train VAE on collected trajectories
    if buffer.episodes_in_buffer > 5:
        print(f"\nTraining VAE on collected trajectories...")
        pivotal_states = vae_system.train(buffer, num_epochs=50, batch_size=4)
        print(f"Discovered {len(pivotal_states)} pivotal states: {pivotal_states[:5]}...")
    
    return policy, buffer, vae_system

def test_goal_policy_with_improved_curiosity():
    """Test Goal-Conditioned Policy with improved curiosity balance."""
    print("Testing Goal-Conditioned Policy with Improved Curiosity Balance:")
    print("=" * 70)
    
    # Create maze environment
    env = MinigridWrapper(
        size=EnvSizes.SMALL,
        mode=EnvModes.MULTIGOAL
    )
    env.phase = 1
    env.randomgen = True
    
    # Create goal-conditioned policy
    policy = GoalConditionedPolicy(lr=1e-3)
    
    # Create VAE system for curiosity rewards
    vae_system = VAESystem(
        state_dim=16,
        action_vocab_size=7, 
        mu0=3.0,
        grid_size=env.size
    )
    
    # Create buffer to store trajectories
    buffer = StatBuffer()
    
    print(f"Environment size: {env.size}x{env.size}")
    print(f"Valid position range: (1,1) to ({env.size-2},{env.size-2})")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print(f"VAE parameters: {sum(p.numel() for p in vae_system.parameters())}")
    
    # Training loop with improved balance
    num_training_episodes = 20
    success_count = 0
    error_count = 0
    
    for episode in range(num_training_episodes):
        # Reset environment
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        # Validate and fix start position if needed
        if not (1 <= start_pos[0] <= env.size-2 and 1 <= start_pos[1] <= env.size-2):
            start_pos = (env.size//2, env.size//2)
            env.agent_pos = start_pos
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Start position: {start_pos}")
        
        # Check reachable positions
        try:
            reachable_positions = env.BFS_all_reachable(start_pos)
            print(f"  Reachable positions: {len(reachable_positions)}")
            
            if len(reachable_positions) < 2:
                print("  Skipping - not enough reachable positions")
                continue
                
        except Exception as e:
            print(f"  Error finding reachable positions: {e}")
            error_count += 1
            continue
        
        # Train one episode with improved curiosity balance
        try:
            # Improved curiosity weight schedule: start lower, decay slower
            curiosity_weight = max(0.3, 1.0 - (episode / 25))  # Decay from 1.0 to 0.3 over 25 episodes
            
            states, actions, rewards, goal_reached = policy.train_goal_policy_episode(
                env, start_pos, 
                max_episode_length=30,  # Reduced episode length
                vae_system=vae_system,
                curiosity_weight=curiosity_weight
            )
            
            print(f"  Episode length: {len(actions)}")
            print(f"  Total reward: {sum(rewards):.3f}")
            print(f"  Average reward per step: {sum(rewards)/len(rewards):.3f}")
            print(f"  Goal reached: {goal_reached}")
            print(f"  Curiosity weight: {curiosity_weight:.2f}")
            
            if goal_reached:
                success_count += 1
            
            # Store trajectory in buffer for VAE training
            for i, ((state, goal), action, reward) in enumerate(zip(states, actions, rewards)):
                next_state = states[i+1][0] if i+1 < len(states) else state
                done = (i == len(states) - 1)
                buffer.add(state, action, reward, next_state, done)
            
            buffer.end_episode()
            
        except Exception as e:
            print(f"  Error during episode: {e}")
            error_count += 1
            continue
    
    print(f"\nImproved Curiosity Training Summary:")
    print(f"Episodes attempted: {num_training_episodes}")
    print(f"Episodes with errors: {error_count}")
    print(f"Episodes completed: {num_training_episodes - error_count}")
    print(f"Goals reached: {success_count}/{num_training_episodes - error_count}")
    if (num_training_episodes - error_count) > 0:
        print(f"Success rate: {success_count/(num_training_episodes - error_count)*100:.1f}%")
    print(f"Trajectories collected: {buffer.episodes_in_buffer}")
    print(f"Total steps: {buffer.total_steps_in_buffer}")
    
    # Train VAE on collected trajectories
    if buffer.episodes_in_buffer > 5:
        print(f"\nTraining VAE on collected trajectories...")
        pivotal_states = vae_system.train(buffer, num_epochs=30, batch_size=3)
        print(f"Discovered {len(pivotal_states)} pivotal states")
        
        # Validate discovered pivotal states
        valid_pivotal = [(x,y) for (x,y) in pivotal_states if 1 <= x <= env.size-2 and 1 <= y <= env.size-2]
        print(f"Valid pivotal states: {len(valid_pivotal)}/{len(pivotal_states)}")
        print(f"Sample valid pivotal states: {valid_pivotal[:5]}")
    
    return policy, buffer, vae_system

def test_alternating_training():
    """Test the complete alternating training framework."""
    print("Testing Complete Alternating Training Framework:")
    print("=" * 60)
    
    # Setup
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 1
    env.randomgen = True
    
    policy = GoalConditionedPolicy(lr=1e-3)
    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=3.0, grid_size=env.size)
    buffer = StatBuffer()
    
    print(f"Environment: {env.size}x{env.size} maze")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print(f"VAE parameters: {sum(p.numel() for p in vae_system.parameters())}")
    
    # Run alternating training
    final_pivotal_states = alternating_training_loop(
        env=env,
        policy=policy, 
        vae_system=vae_system,
        buffer=buffer,
        max_iterations=6
    )
    
    # Test final policy performance
    print(f"\n" + "="*60)
    print("Testing Final Policy Performance:")
    
    success_count = 0
    test_episodes = 10
    
    for test in range(test_episodes):
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        try:
            states, actions, rewards, goal_reached = policy.train_goal_policy_episode(
                env, start_pos, max_episode_length=20, vae_system=vae_system, curiosity_weight=0.1
            )
            
            if goal_reached:
                success_count += 1
                print(f"Test {test+1}: SUCCESS in {len(actions)} steps")
            else:
                print(f"Test {test+1}: Failed - {sum(rewards):.2f} total reward")
                
        except Exception as e:
            print(f"Test {test+1}: Error - {e}")
    
    print(f"\nFinal Performance Summary:")
    print(f"Success rate: {success_count}/{test_episodes} ({success_count/test_episodes*100:.1f}%)")
    print(f"Discovered pivotal states: {len(final_pivotal_states)}")
    print(f"Total training episodes: {buffer.episodes_in_buffer}")
    
    return policy, vae_system, final_pivotal_states, buffer

def test_graph_visualizer():
    """Test the visualizer with sample data."""
    
    # Create sample graph
    graph = GraphManager()
    
    # Add nodes (coordinates)
    nodes = [(1, 1), (3, 2), (5, 1), (2, 4), (4, 4), (6, 3)]
    for node in nodes:
        graph.add_node(node)
    
    # Add edges
    edges = [
        ((1, 1), (3, 2), 2),
        ((3, 2), (2, 4), 3),
        ((3, 2), (5, 1), 2),
        ((5, 1), (6, 3), 3),
        ((2, 4), (4, 4), 2),
        ((4, 4), (6, 3), 2)
    ]
    for start, end, weight in edges:
        graph.add_edge(start, end, weight)
    
    # Visualize
    viz = GraphVisualizer(graph)
    
    # Basic visualization
    fig1, ax1 = viz.visualize()
    plt.show()
    
    # Show statistics
    viz.show_statistics()
    
    # Test shortest path visualization
    path, distance = graph.shortest_path((1, 1), (6, 3))
    if path:
        fig2, ax2 = viz.show_path(path)
        plt.show()
        print(f"Shortest path from (1,1) to (6,3): {path}, distance: {distance}")

# GRAPH UPDATED TESTS MECHANISTICS -----------------------------------------

def test_edge_action_storage():
    """Test that edge refinement stores action sequences correctly."""
    print("Testing Edge Action Storage:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    from local_networks.policy_networks import GoalConditionedPolicy
    from utils.graph_manager import GraphManager
    
    # Create environment
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 1
    env.randomgen = True
    
    # Create policy
    policy = GoalConditionedPolicy(lr=5e-3, verbose=True)
    
    # Sample pivotal states close together for easy testing
    pivotal_states = [(2, 2), (5, 5)]
    
    print(f"Testing edge discovery between: {pivotal_states}")
    
    # Discover edges
    raw_edges = policy.discover_edges_between_pivotal_states(
        env, pivotal_states, max_walk_length=15, num_attempts=5
    )
    
    print(f"\nDiscovered {len(raw_edges)} raw edges")
    for (start, end), path in raw_edges.items():
        print(f"  {start} -> {end}: {len(path)} positions")
    
    # Refine with policy (should now store actions)
    refined_edges = policy.refine_paths_with_goal_policy(env, raw_edges)
    
    print(f"\nRefined edges with action sequences:")
    for (start, end), (path, weight, actions) in refined_edges.items():
        print(f"  {start} -> {end}:")
        print(f"    Path length: {len(path)} positions")
        print(f"    Weight: {weight}")
        print(f"    Actions: {len(actions)} actions: {actions}")
        
        # Verify action count matches path
        expected_actions = len(path) - 1
        if len(actions) == expected_actions:
            print(f"    ✓ Action count correct ({len(actions)} == {expected_actions})")
        else:
            print(f"    ✗ Action count mismatch! Got {len(actions)}, expected {expected_actions}")
    
    print("\n" + "=" * 60)
    return refined_edges

def test_graph_manager_action_storage():
    """Test that GraphManager stores and retrieves actions correctly."""
    print("Testing GraphManager Action Storage:")
    print("=" * 60)
    
    from utils.graph_manager import GraphManager
    
    # Create graph
    graph = GraphManager()
    
    # Add nodes
    nodes = [(2, 2), (5, 5), (8, 8)]
    for node in nodes:
        graph.add_node(node)
    
    # Add edges WITH action sequences
    test_actions_1 = [2, 2, 1, 2, 2]  # move, move, turn_right, move, move
    test_actions_2 = [2, 0, 2, 2]     # move, turn_left, move, move
    
    graph.add_edge((2, 2), (5, 5), weight=5, action_sequence=test_actions_1)
    graph.add_edge((5, 5), (8, 8), weight=4, action_sequence=test_actions_2)
    
    print(f"Added {len(graph.nodes)} nodes")
    print(f"Added {len(graph.edges)} edges with actions")
    
    # Retrieve actions
    print("\nRetrieving action sequences:")
    
    actions_1 = graph.get_edge_actions((2, 2), (5, 5))
    print(f"  (2,2) -> (5,5): {actions_1}")
    assert actions_1 == test_actions_1, "Action retrieval failed!"
    print("    ✓ Correct")
    
    actions_2 = graph.get_edge_actions((5, 5), (8, 8))
    print(f"  (5,5) -> (8,8): {actions_2}")
    assert actions_2 == test_actions_2, "Action retrieval failed!"
    print("    ✓ Correct")
    
    # Test non-existent edge
    actions_none = graph.get_edge_actions((2, 2), (8, 8))
    print(f"  Non-existent edge: {actions_none}")
    assert actions_none is None, "Should return None for non-existent edge!"
    print("    ✓ Correct (None)")
    
    print("\n" + "=" * 60)
    return graph

def test_worker_action_execution():
    """Test that Worker executes stored actions during traversal."""
    print("Testing Worker Action Execution:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    from local_networks.hierarchical_system import HierarchicalWorker
    from utils.graph_manager import GraphManager
    
    # Create graph with known actions
    graph = GraphManager()
    pivotal_states = [(2, 2), (5, 5)]
    
    for state in pivotal_states:
        graph.add_node(state)
    
    # Manually create a simple action sequence
    # For testing: just move forward 3 times
    test_actions = [2, 2, 2]  # 3x move_forward
    graph.add_edge((2, 2), (5, 5), weight=3, action_sequence=test_actions)
    
    # Create worker
    worker = HierarchicalWorker(graph, pivotal_states, verbose=True)
    
    print(f"Created worker with graph:")
    print(f"  Nodes: {graph.nodes}")
    print(f"  Edge (2,2)->(5,5) actions: {graph.get_edge_actions((2,2), (5,5))}")
    
    # Simulate traversal
    print(f"\nSimulating traversal from (2,2) to (5,5):")
    
    current_state = (2, 2)
    wide_goal = (5, 5)
    narrow_goal = (5, 5)
    
    # Check if traversal should be initiated
    should_traverse = worker.should_traverse(current_state, wide_goal)
    print(f"  Should traverse: {should_traverse}")
    
    if should_traverse:
        # Plan traversal
        path = worker.plan_traversal(current_state, wide_goal)
        print(f"  Planned path: {path}")
        
        # Manually set traversal path (simulating what get_action does)
        worker.current_traversal_path = path
        worker.traversal_step = 0
        
        # Execute traversal steps
        executed_actions = []
        for step in range(len(path) - 1):
            action, log_prob, value = worker.get_action(current_state, wide_goal, narrow_goal)
            executed_actions.append(action)
            print(f"  Step {step+1}: action={action} (expected={test_actions[step] if step < len(test_actions) else 'N/A'})")
            
            # Check if action matches
            if step < len(test_actions):
                if action == test_actions[step]:
                    print(f"    ✓ Correct action")
                else:
                    print(f"    ✗ WRONG action! Expected {test_actions[step]}, got {action}")
        
        print(f"\n  Executed actions: {executed_actions}")
        print(f"  Expected actions: {test_actions}")
        
        if executed_actions == test_actions:
            print("  ✓ ALL ACTIONS CORRECT!")
        else:
            print("  ✗ Action sequence mismatch")
    else:
        print("  ✗ Traversal not initiated (shouldn't happen)")
    
    print("\n" + "=" * 60)

def test_full_traversal_in_environment():
    """Test complete traversal flow in actual environment."""
    print("Testing Full Traversal in Environment:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    from local_networks.policy_networks import GoalConditionedPolicy
    from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker
    import traceback
    
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 1
    env.randomgen = True
    
    print("Quick Phase 1 to get pivotal states and graph...")
    policy = GoalConditionedPolicy(lr=5e-3)
    
    pivotal_states = [(2, 2), (5, 5), (7, 7)]
    
    try:
        print("\nStep 1: Discovering edges...")
        raw_edges = policy.discover_edges_between_pivotal_states(
            env, pivotal_states, max_walk_length=20, num_attempts=3
        )
        print(f"  Found {len(raw_edges)} raw edges")
        
        print("\nStep 2: Refining edges...")
        refined_edges = policy.refine_paths_with_goal_policy(env, raw_edges)
        print(f"  Refined {len(refined_edges)} edges")
        
        # DEBUG: Verify structure
        print("\nStep 3: Verifying refined_edges structure...")
        for key, value in refined_edges.items():
            print(f"  Edge {key}:")
            print(f"    Type: {type(value)}")
            print(f"    Length: {len(value)}")
            if len(value) == 3:
                path, weight, actions = value
                print(f"    ✓ Correct 3-tuple: path={len(path)}, weight={weight}, actions={len(actions)}")
            else:
                print(f"    ✗ WRONG! Expected 3-tuple, got: {value}")
        
        print("\nStep 4: Constructing world graph...")
        world_graph = policy.construct_world_graph(pivotal_states, refined_edges)
        print(f"  ✓ Graph constructed successfully")
        
        print("\nStep 5: Creating hierarchical system...")
        manager = HierarchicalManager(pivotal_states, neighborhood_size=3, horizon=10)
        worker = HierarchicalWorker(world_graph, pivotal_states, verbose=True)
        
        manager.initialize_from_goal_policy(policy)
        worker.initialize_from_goal_policy(policy)
        
        print("  ✓ Hierarchical system created")
        
    except Exception as e:
        print(f"\n✗ ERROR at some step:")
        print(f"  Message: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 60)

def run_all_traversal_tests():
    """Run all traversal tests in sequence."""
    print("\n" + "#" * 60)
    print("RUNNING ALL GRAPH TRAVERSAL TESTS")
    print("#" * 60 + "\n")
    
    try:
        print("Test 1: Edge Action Storage")
        refined_edges = test_edge_action_storage()
        print("\n✓ Test 1 passed\n")
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}\n")
        return
    
    try:
        print("Test 2: GraphManager Action Storage")
        graph = test_graph_manager_action_storage()
        print("\n✓ Test 2 passed\n")
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}\n")
        return
    
    try:
        print("Test 3: Worker Action Execution")
        test_worker_action_execution()
        print("\n✓ Test 3 passed\n")
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}\n")
        return
    
    try:
        print("Test 4: Full Traversal in Environment")
        test_full_traversal_in_environment()
        print("\n✓ Test 4 passed\n")
    except Exception as e:
        print(f"\n✗ Test 4 failed: {e}\n")
        return
    
    print("\n" + "#" * 60)
    print("ALL TESTS PASSED!")
    print("#" * 60)

# GRAPH UPDATE TESTS INTEGRATION -------------------------------------------------

def test_ball_collection_mechanics():
    """Test that balls are collected automatically when stepping on them."""
    print("Testing Ball Collection Mechanics:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 2  # Phase 2 has balls
    env.randomgen = True
    
    obs = env.reset()
    print(f"Environment reset in Phase 2")
    print(f"Agent position: {env.agent_pos}")
    print(f"Total balls: {env.total_balls}")
    print(f"Active balls: {len(env.active_balls)}")
    print(f"Ball positions: {list(env.active_balls)[:3]}...")
    
    # Find a ball and navigate to it
    if len(env.active_balls) > 0:
        target_ball = list(env.active_balls)[0]
        print(f"\nTarget ball at: {target_ball}")
        
        initial_balls = len(env.active_balls)
        
        # Navigate toward ball (simplified - just demonstrate mechanics)
        collected = False
        for step in range(50):
            # Simple navigation toward ball
            current_pos = tuple(env.agent_pos)
            
            # Determine action to move toward ball
            dx = target_ball[0] - current_pos[0]
            dy = target_ball[1] - current_pos[1]
            
            if abs(dx) > abs(dy):
                action = 2  # move_forward (assuming facing right direction)
            else:
                action = 2
            
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = tuple(env.agent_pos)
            
            # Check if we collected the ball
            if new_pos == target_ball and target_ball not in env.active_balls:
                print(f"  Step {step+1}: Ball collected!")
                print(f"    Position: {new_pos}")
                print(f"    Reward: {reward}")
                print(f"    Active balls remaining: {len(env.active_balls)}")
                collected = True
                break
            
            if new_pos != current_pos:
                print(f"  Step {step+1}: Moved to {new_pos}, distance to ball: {abs(dx) + abs(dy)}")
        
        if collected:
            print("\n✓ Ball collection mechanics working!")
        else:
            print("\n✗ Could not collect ball (may need better navigation)")
    else:
        print("✗ No balls spawned in Phase 2!")
    
    print("\n" + "=" * 60)

def test_phase_switching():
    """Test Phase 1 (empty) vs Phase 2 (with balls) switching."""
    print("Testing Phase Switching:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    
    # Test Phase 1
    print("Phase 1 (empty maze for graph discovery):")
    env.phase = 1
    env.randomgen = True
    obs = env.reset()
    
    print(f"  Agent position: {env.agent_pos}")
    print(f"  Active balls: {len(env.active_balls)}")
    
    # Count items in grid
    items_phase1 = 0
    for x in range(1, env.size-1):
        for y in range(1, env.size-1):
            cell = env.grid.get(x, y)
            if cell is not None and not isinstance(cell, type(env.grid.get(0, 0))):
                items_phase1 += 1
    
    print(f"  Items in maze: {items_phase1}")
    assert env.total_balls == 0, "Phase 1 should have no balls!"
    print("  ✓ Phase 1: Empty maze as expected")
    
    # Test Phase 2
    print("\nPhase 2 (with balls for task):")
    env.phase = 2
    env.randomgen = True
    obs = env.reset()
    
    print(f"  Agent position: {env.agent_pos}")
    print(f"  Total balls: {env.total_balls}")
    print(f"  Active balls: {len(env.active_balls)}")
    
    assert env.total_balls > 0, "Phase 2 should have balls!"
    assert len(env.active_balls) > 0, "Phase 2 should have active balls!"
    print("  ✓ Phase 2: Balls spawned as expected")
    
    print("\n" + "=" * 60)

def test_worker_navigation_in_maze():
    """Test Worker can actually navigate in the real maze."""
    print("Testing Worker Navigation in Real Maze:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    from local_networks.policy_networks import GoalConditionedPolicy
    from local_networks.hierarchical_system import HierarchicalWorker
    from utils.graph_manager import GraphManager
    
    # Create environment
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 1
    env.randomgen = True
    obs = env.reset()
    
    # Create simple graph
    graph = GraphManager()
    pivotal_states = [(2, 2), (5, 5)]
    for state in pivotal_states:
        graph.add_node(state)
    
    # Create worker
    worker = HierarchicalWorker(graph, pivotal_states, verbose=True)
    policy = GoalConditionedPolicy(lr=5e-3)
    worker.initialize_from_goal_policy(policy)
    
    # Place agent at start
    env.agent_pos = (2, 2)
    env.agent_dir = 0
    
    print(f"Starting navigation test:")
    print(f"  Start: {tuple(env.agent_pos)}")
    print(f"  Goal: (5, 5)")
    
    # Navigate using worker policy
    for step in range(30):
        current_pos = tuple(env.agent_pos)
        wide_goal = (5, 5)
        narrow_goal = (5, 5)
        
        # Get action from worker
        action, log_prob, value = worker.get_action(current_pos, wide_goal, narrow_goal)
        
        action_names = ['turn_left', 'turn_right', 'move_forward']
        print(f"  Step {step+1}: pos={current_pos}, action={action_names[action]}")
        
        # Take step
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = tuple(env.agent_pos)
            
            if new_pos != current_pos:
                print(f"    -> moved to {new_pos}")
            
            # Check if reached goal
            if new_pos == wide_goal:
                print(f"  ✓ GOAL REACHED in {step+1} steps!")
                break
            
            if terminated or truncated:
                print(f"  Episode ended")
                break
                
        except (AssertionError, IndexError) as e:
            print(f"    Action failed (out of bounds): {e}")
            continue
    else:
        print(f"  ✗ Did not reach goal in 30 steps")
    
    print("\n" + "=" * 60)

def test_manager_worker_coordination(verb):
    """Test Manager and Worker work together in environment."""
    print("Testing Manager-Worker Coordination:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    from local_networks.policy_networks import GoalConditionedPolicy
    from local_networks.hierarchical_system import HierarchicalManager, HierarchicalWorker
    from utils.graph_manager import GraphManager
    
    # Setup
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 1
    env.randomgen = True
    obs = env.reset()
    
    # Create simple graph
    graph = GraphManager()
    pivotal_states = [(2, 2), (4, 4), (6, 6)]
    for state in pivotal_states:
        graph.add_node(state)
    
    # Create hierarchical system
    manager = HierarchicalManager(pivotal_states, neighborhood_size=3, horizon=10)
    worker = HierarchicalWorker(graph, pivotal_states, verbose=False)
    
    policy = GoalConditionedPolicy(lr=5e-3)
    manager.initialize_from_goal_policy(policy)
    worker.initialize_from_goal_policy(policy)
    
    # Place agent
    env.agent_pos = (2, 2)
    env.agent_dir = 0
    
    print(f"Testing Manager-Worker coordination:")
    print(f"  Pivotal states: {pivotal_states}")
    print(f"  Starting at: {tuple(env.agent_pos)}")
    
    # Simulate one horizon
    current_state = tuple(env.agent_pos)
    
    # Manager selects goals
    print(f"\nManager decision:")
    wide_goal, narrow_goal, manager_log_prob, manager_value = manager.get_manager_action(current_state)
    print(f"  Wide goal (pivotal): {wide_goal}")
    print(f"  Narrow goal (local): {narrow_goal}")
    print(f"  Manager value estimate: {manager_value.item():.3f}")
    
    # Worker executes
    print(f"\nWorker execution (horizon=10):")
    horizon_reward = 0
    
    for h in range(10):
        current_pos = tuple(env.agent_pos)
        
        # Worker action
        action, worker_log_prob, worker_value = worker.get_action(
            current_pos, wide_goal, narrow_goal
        )
        
        action_names = ['turn_left', 'turn_right', 'move_forward']
        
        # Environment step
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            new_pos = tuple(env.agent_pos)
            
            horizon_reward += reward
            
            if new_pos != current_pos or h % 3 == 0:
                print(f"  Step {h+1}: {current_pos} -> {action_names[action]} -> {new_pos}, reward={reward:.3f}")
            
            # Check goal reached
            worker_reward = worker.compute_reward(new_pos, wide_goal, narrow_goal)
            if worker_reward > 0:
                print(f"  ✓ Worker reached goal! Internal reward: {worker_reward}")
                break
            
            if terminated or truncated:
                break
                
        except (AssertionError, IndexError):
            continue
    
    print(f"\nHorizon summary:")
    print(f"  Total horizon reward: {horizon_reward:.3f}")
    print(f"  Manager receives this as feedback")
    
    print("\n" + "=" * 60)

def run_environment_integration_tests():
    """Run all environment integration tests."""
    print("\n" + "#" * 60)
    print("ENVIRONMENT INTEGRATION TESTS")
    print("#" * 60 + "\n")
    
    tests = [
        ("Phase Switching", test_phase_switching),
        ("Ball Collection", test_ball_collection_mechanics),
        ("Worker Navigation", test_worker_navigation_in_maze),
        ("Manager-Worker Coordination", test_manager_worker_coordination),
        #("Complete Episode", test_complete_episode_with_balls),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print('='*60)
            test_func()
            print(f"\n✓ {name} PASSED\n")
            passed += 1
        except Exception as e:
            import traceback
            print(f"\n✗ {name} FAILED:")
            print(f"  Error: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("#" * 60)

def test_phase2_ball_spawning():
    """Debug test: Do balls spawn during Phase 2 training?"""
    env = MinigridWrapper(size=EnvSizes.MEDIUM, mode=EnvModes.MULTIGOAL)
    env.phase = 2
    env.randomgen = True
    
    for i in range(5):
        obs = env.reset()
        print(f"Reset {i+1}: {len(env.active_balls)} balls at {list(env.active_balls)[:3]}")
# COMPREHENSIVE TEST SUITE ----------------------------------------------------

def run_all_comprehensive_tests():
    """Run all tests - traversal mechanics + environment integration."""
    print("\n" + "#" * 70)
    print("COMPREHENSIVE TEST SUITE")
    print("#" * 70 + "\n")
    
    results = {
        'traversal': {'passed': 0, 'failed': 0, 'tests': []},
        'environment': {'passed': 0, 'failed': 0, 'tests': []}
    }
    
    # Part 1: Graph Traversal Tests
    print("\n" + "=" * 70)
    print("PART 1: GRAPH TRAVERSAL MECHANICS")
    print("=" * 70 + "\n")
    
    traversal_tests = [
        ("Edge Action Storage", test_edge_action_storage),
        ("GraphManager Storage", test_graph_manager_action_storage),
        ("Worker Action Execution", test_worker_action_execution),
        ("Full Traversal Integration", test_full_traversal_in_environment),
    ]
    
    for name, test_func in traversal_tests:
        try:
            print(f"\nRunning: {name}")
            print("-" * 60)
            test_func()
            print(f"✓ {name} PASSED")
            results['traversal']['passed'] += 1
            results['traversal']['tests'].append((name, 'PASS'))
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            results['traversal']['failed'] += 1
            results['traversal']['tests'].append((name, 'FAIL', str(e)))
    
    # Part 2: Environment Integration Tests
    print("\n" + "=" * 70)
    print("PART 2: ENVIRONMENT INTEGRATION")
    print("=" * 70 + "\n")
    
    env_tests = [
        ("Phase Switching", test_phase_switching),
        ("Ball Collection", test_ball_collection_mechanics),
        ("Worker Navigation", test_worker_navigation_in_maze),
        ("Manager-Worker Coordination", test_manager_worker_coordination),
        #("Complete Episode", test_complete_episode_with_balls),
    ]
    
    for name, test_func in env_tests:
        try:
            print(f"\nRunning: {name}")
            print("-" * 60)
            test_func()
            print(f"✓ {name} PASSED")
            results['environment']['passed'] += 1
            results['environment']['tests'].append((name, 'PASS'))
        except Exception as e:
            import traceback
            print(f"✗ {name} FAILED: {e}")
            # Print short traceback
            traceback.print_exc()
            results['environment']['failed'] += 1
            results['environment']['tests'].append((name, 'FAIL', str(e)))
    
    # Final Summary
    print("\n" + "#" * 70)
    print("FINAL TEST SUMMARY")
    print("#" * 70)
    
    total_passed = results['traversal']['passed'] + results['environment']['passed']
    total_failed = results['traversal']['failed'] + results['environment']['failed']
    total_tests = total_passed + total_failed
    
    print(f"\nGraph Traversal: {results['traversal']['passed']}/{results['traversal']['passed'] + results['traversal']['failed']} passed")
    for test in results['traversal']['tests']:
        status = "✓" if test[1] == 'PASS' else "✗"
        print(f"  {status} {test[0]}")
        if test[1] == 'FAIL':
            print(f"    Error: {test[2][:80]}...")
    
    print(f"\nEnvironment Integration: {results['environment']['passed']}/{results['environment']['passed'] + results['environment']['failed']} passed")
    for test in results['environment']['tests']:
        status = "✓" if test[1] == 'PASS' else "✗"
        print(f"  {status} {test[0]}")
        if test[1] == 'FAIL':
            print(f"    Error: {test[2][:80]}...")
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    print(f"{'='*70}")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! Ready for training.")
    else:
        print(f"\n⚠️  {total_failed} test(s) failed. Review errors above.")
    
    return results

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
    
    connectivity_pct = 100 * connected_pairs / total_pairs
    print(f"  Connected pairs: {connected_pairs}/{total_pairs} ({connectivity_pct:.1f}%)")
    
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

def test_phase1_with_diagnostics(config=None):
    """
    Test Phase 1 using alternating_training_loop with diagnostic tracking.
    """
    default_config = {
        'maze_size': EnvSizes.SMALL,
        'phase1_iterations': 8,
        'vae_mu0': 10.0,
        'goal_policy_lr': 5e-4,
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
    env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL)
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

externalconfig = {
        'maze_size': EnvSizes.SMALL,
        'phase1_iterations': 8,
        'phase2_episodes': 100,
        'max_steps_per_episode': 500,
        'manager_horizon': 20,
        'neighborhood_size': 5,
        'manager_lr': 1e-4,
        'worker_lr': 5e-3,
        'vae_mu0': 10.0,
        'diagnostic_interval': 50000,  # NEW: Print diagnostics every K steps
        'diagnostic_checkstart': True,  # NEW: Print every step for first 15 steps
        'full_breakdown_every': 50,  # NEW: Full breakdown every N episodes
        'device': 'cuda'  # <-- ADD THIS: 'cpu' or 'cuda'
    }

def train_full_phase1_phase2(config=externalconfig):
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
    env = MinigridWrapper(size=config['maze_size'], mode=EnvModes.MULTIGOAL, max_steps=config['max_steps_per_episode'])
    env.phase = 1
    env.randomgen = True
    
    policy = GoalConditionedPolicy(lr=config['manager_lr'],device=config['device'])
    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=config['vae_mu0'], grid_size=env.size)
    buffer = StatBuffer()
    
    print("\nPHASE 1: World Graph Discovery")
    start_time = time.time()
    
    pivotal_states, world_graph = alternating_training_loop(
        env, policy, vae_system, buffer, max_iterations=config['phase1_iterations']
    )
    
    phase1_time = time.time() - start_time
    print(f"\nPhase 1 complete in {phase1_time:.1f}s")
    print(f"  Pivotal states: {len(pivotal_states)}")
    print(f"  Graph edges: {len(world_graph.edges)}")
    
    # Diagnose graph connectivity
    reachable, unreachable = diagnose_graph_connectivity(
        world_graph, pivotal_states, env
    )

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
        lr=config['worker_lr'],  # ← ADD THIS
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
    diagnostic_checkstart=config['diagnostic_checkstart']  # NEW
    )
    
    print("\nPHASE 2: Hierarchical Training")
    
    # Tracking
    metrics = {
        'rewards': [],
        'steps': [],
        'manager_updates': [],
        'worker_updates': [],
        'traversals': [],
        'times': [],
        'optimal_rewards': []  # NEW: Add this line
    }
    
    for episode in range(config['phase2_episodes']):
        ep_start = time.time()
        if episode%10 == 0 and episode > 0:
            print(f"\n--- Episode {episode+1}/{config['phase2_episodes']} ---")
        stats = trainer.train_episode(max_steps=config['max_steps_per_episode'],full_breakdown_every=config['full_breakdown_every'])
        
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

def main():
    # pygame.init()
    # env = MinigridWrapper(render_mode="human",mode=EnvModes.MULTIGOAL, phase_one_eps=10)
    # #env = FastWrapper("MiniGrid-KeyCorridorS3R2-v0",1000,"human")
    # print(os.path.dirname(minigrid.__file__))

    # enable manual control for testing
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()

    #train_full_phase1_phase2()
    # Compare maze sizes
    # Compare different mu0 values
    runs = {}
    for mu0 in [3.0, 5.0, 10.0, 15.0]:
        config = {'vae_mu0': mu0, 'phase1_iterations': 50, 'maze_size': EnvSizes.MEDIUM}
        runs[f'mu0={mu0}'] = test_phase1_with_diagnostics(config)

    # compare_phase1_runs(runs)

    # runs = {}
    # for size in [EnvSizes.SMALL, EnvSizes.MEDIUM]:
    #     config = {'maze_size': size}
    #     runs[f'{size.name}'] = test_phase1_with_diagnostics(config)

    # compare_phase1_runs(runs)

if __name__ == "__main__":
    main()
