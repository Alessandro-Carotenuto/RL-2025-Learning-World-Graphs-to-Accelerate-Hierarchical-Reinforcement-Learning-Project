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
from local_networks.hierarchical_system import hierarchical_system_tests

#---------------------------------------------------------------------------------------
# MAIN ALTERNATING TRAINING LOOP - UPDATED TO INCLUDE WORLD GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------------------

def alternating_training_loop(env, policy, vae_system, buffer, max_iterations: int = 8):
    """
    Main alternating training loop - UPDATED to include world graph construction.
    """
    print("Starting Alternating Training Loop:")
    print("=" * 50)
    
    reconstruction_losses = []
    all_pivotal_states = []
    pivotal_states = []
    
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        
        # Phase 1: Train VAE on current trajectory buffer
        print(f"Phase 1: Training VAE on {buffer.episodes_in_buffer} episodes...")
        
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
        
        # Train VAE and get pivotal states
        try:
            pivotal_states = vae_system.train(buffer, num_epochs=25, batch_size=3)
        except Exception as e:
            print(f"VAE training failed: {e}")
            continue
        
        # Track reconstruction loss for plateau detection
        if vae_system.training_history:
            current_loss = vae_system.training_history[-1]['reconstruction_loss']
            reconstruction_losses.append(current_loss)
            print(f"Current reconstruction loss: {current_loss:.4f}")
        
        all_pivotal_states.append(pivotal_states.copy())
        print(f"Discovered {len(pivotal_states)} pivotal states: {pivotal_states[:3]}...")
        
        # Phase 2: Collect trajectories starting from pivotal states
        print(f"Phase 2: Collecting trajectories from top {min(5, len(pivotal_states))} pivotal states...")
        
        episodes_collected = 0
        for i, start_state in enumerate(pivotal_states[:5]):
            print(f"  Collecting from pivotal state {i+1}: {start_state}")
            
            curiosity_weight = max(0.2, 0.8 - (iteration * 0.1))
            
            try:
                episodes = policy.collect_episodes_from_position(
                    env, start_state,
                    num_episodes=2,
                    vae_system=vae_system,
                    curiosity_weight=curiosity_weight
                )
                
                if episodes:
                    buffer.add_episodes(episodes)
                    episodes_collected += len(episodes)
            except Exception as e:
                print(f"  Failed to collect from {start_state}: {e}")
                continue
        
        print(f"  Collected {episodes_collected} new episodes")
        print(f"  Total episodes in buffer: {buffer.episodes_in_buffer}")
        
        # Phase 3: Check for plateau (convergence)
        if len(reconstruction_losses) >= 3:
            recent_losses = reconstruction_losses[-3:]
            loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) for i in range(1, len(recent_losses))]
            avg_change = sum(loss_changes) / len(loss_changes)
            
            print(f"Average loss change over last 3 iterations: {avg_change:.5f}")
            
            if avg_change < 0.005:
                print("Reconstruction loss has plateaued - training converged!")
                break
        
        # Phase 4: Optional diversity collection
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
    
    # PHASE 1 COMPLETION: Construct world graph
    world_graph = policy.complete_world_graph_discovery(env, pivotal_states)
    
    # Final summary
    print(f"\nAlternating Training Complete!")
    print(f"Total iterations: {len(reconstruction_losses)}")
    print(f"Final episodes in buffer: {buffer.episodes_in_buffer}")
    print(f"Final pivotal states ({len(pivotal_states)}): {pivotal_states}")
    
    if len(all_pivotal_states) > 1:
        print(f"\nPivotal state progression:")
        for i, states in enumerate(all_pivotal_states):
            print(f"  Iteration {i+1}: {len(states)} states - {states[:3]}...")
    
    return pivotal_states, world_graph

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
        grid_size=24
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
    """Test the GoalConditionedPolicy implementation."""
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
    env.randomgen = False
    
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
    env.randomgen = False
    
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
    env.randomgen = False
    
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
    env.randomgen = False
    
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


def main():
    # pygame.init()
    # env = MinigridWrapper(render_mode="human",mode=EnvModes.MULTIGOAL, phase_one_eps=10)
    # #env = FastWrapper("MiniGrid-KeyCorridorS3R2-v0",1000,"human")
    # print(os.path.dirname(minigrid.__file__))

    # # enable manual control for testing
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()
    # fake_ep_data=TestBuffer()
    # v=Visualizer(fake_ep_data)
    # v.total_reward_stream()
    # v.total_reward_by_ep_stream()
    # v.reward_stream_for_ep(10)
    
    hierarchical_system_tests()

    input()
    
    
    

if __name__ == "__main__":
    main()
