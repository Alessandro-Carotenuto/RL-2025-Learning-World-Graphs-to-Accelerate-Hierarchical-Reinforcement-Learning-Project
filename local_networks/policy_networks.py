import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from typing import List, Tuple, Dict, Optional

from utils.misc import manhattan_distance,sample_goal_position
from utils.graph_manager import GraphManager

old_diag_phase_1 = False  # Toggle detailed diagnostics in phase 1
#-----------------------------------------------------------------------------

class GoalConditionedPolicy(nn.Module):
    """
    Phase 1: Goal-conditioned policy πg for world graph discovery.
    A2C-based policy that learns to navigate between nearby states.
    """
    
    def __init__(self, lr: float = 5e-3, verbose: bool =False, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.verbose = verbose
        self.device = device
        
        # Neural network components
        # self.net = nn.Sequential(
        #     nn.Linear(4, 64),     # [state_x, state_y, goal_x, goal_y]
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU()
        # ).to(device)
        
        # LSTM instead of Sequential
        self.lstm = nn.LSTM(
            input_size=4,  # [state_x, state_y, goal_x, goal_y]
            hidden_size=64,
            num_layers=1,
            batch_first=True
        ).to(device)

         # LSTM hidden state
        self.hidden_state = None


        self.actor = nn.Linear(64, 7).to(device)    # 7 MiniGrid actions
        self.critic = nn.Linear(64, 1).to(device)   # Value function
        
        # INCREASED LEARNING RATE for better policy updates
        self.optimizer = optim.Adam(self.parameters(), lr=lr)  # Now 5e-3 instead of 1e-3
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.entropy_coef = 0.05    # Entropy regularization
        self.value_coef = 0.5       # Value loss coefficient
        
    def train_goal_policy_episode(self, env, start_pos: Tuple[int, int], 
                        max_episode_length: int = 50,
                        vae_system=None, 
                        curiosity_weight: float = 1.0) -> Tuple[List, List, List, bool]:
        """
        Train the goal-conditioned policy with enhanced reward structure and diagnostics.
        """
        # Sample goal position
        self.hidden_state = None  # Reset LSTM hidden state at episode start
        goal_pos = sample_goal_position(env, start_pos, max_distance=20)
        
        # Initialize episode data
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        
        # Track visited states for curiosity computation
        visited_states = [start_pos]
        
        # Place agent at start position and ensure environment is ready
        obs = env.reset()
        env.agent_pos = start_pos
        env.agent_dir = 0
        current_pos = start_pos
        goal_reached = False
        
        if old_diag_phase_1:
            print(f"    Goal: {goal_pos}, Distance: {manhattan_distance(start_pos, goal_pos)}")
        
        # DIAGNOSTICS: Track policy behavior
        total_goal_reward = 0
        distance_changes = []
        
        episode_trajectory = []  # Store (state, action) pairs
        # Run episode
        for step in range(max_episode_length):
            # Get action from policy
            action, log_prob, value = self.get_action(current_pos, goal_pos)
            episode_trajectory.append((current_pos, action))
            # Store state-action data
            states.append((current_pos, goal_pos))
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            
            # Take environment step
            obs, env_reward, terminated, truncated, info = env.step(action)
            
            # Get new position
            if hasattr(env, 'agent_pos') and env.agent_pos is not None:
                next_pos = tuple(env.agent_pos)
            else:
                next_pos = current_pos
            
            # Update visited states for curiosity computation
            if next_pos != current_pos:
                visited_states.append(next_pos)
            
            # Action name for debug
            action_names = ['turn_left', 'turn_right', 'move_forward', 'pickup', 'drop', 'toggle', 'done']
            action_name = action_names[action] if action < len(action_names) else f'action_{action}'
            
            # ENHANCED REWARD STRUCTURE
            old_distance = manhattan_distance(current_pos, goal_pos)
            new_distance = manhattan_distance(next_pos, goal_pos)
            
            # MAJOR INCREASE: Goal reward now 10x larger
            goal_reward = 10.0 if next_pos == goal_pos else 0.0  # Was 1.0
            
            # Progress reward: reward getting closer to goal
            progress_reward = 0.2 if new_distance < old_distance else 0.0
            
            step_penalty = -0.01
            
            # Curiosity reward computation (reduced since goal reward increased)
            curiosity_reward = 0.0
            if vae_system is not None and len(visited_states) > 1:
                # CORRECT - uses episode_trajectory (state-action pairs) + new method
                window_size = min(5, len(episode_trajectory))
                recent_trajectory = episode_trajectory[-window_size:]
                base_curiosity = vae_system.compute_curiosity_reward_from_trajectory(recent_trajectory)

                curiosity_reward = base_curiosity * curiosity_weight # Scale by weight
                curiosity_reward = min(1.0, curiosity_reward)  # Cap curiosity reward to prevent spikes
            
            total_reward = goal_reward + progress_reward + step_penalty + curiosity_reward
            rewards.append(total_reward)
            
            # DIAGNOSTICS: Track metrics
            total_goal_reward += goal_reward
            distance_changes.append(new_distance - old_distance)
            
            # Enhanced debug output with reward breakdown
            if next_pos != current_pos or step % 10 == 0:
                reward_breakdown = f"(goal:{goal_reward:.1f}, prog:{progress_reward:.2f}, step:{step_penalty:.2f}"
                if curiosity_reward > 0.01:
                    reward_breakdown += f", cur:{curiosity_reward:.3f}"
                reward_breakdown += f") = {total_reward:.2f}"
                if self.verbose:    
                    if old_diag_phase_1:
                        print(f"    Step {step}: {current_pos} -> {action_name} -> {next_pos} {reward_breakdown}")
            
            # Check if goal reached or episode ended
            if goal_reward > 0:
                goal_reached = True
                if old_diag_phase_1:
                    print(f"    GOAL REACHED in {step+1} steps! Total goal reward: {total_goal_reward}")
                break
                
            if terminated or truncated:
                break
                
            current_pos = next_pos
        
        # DIAGNOSTICS: Print episode summary
        avg_distance_change = sum(distance_changes) / len(distance_changes) if distance_changes else 0
        if old_diag_phase_1:
            print(f"    Episode summary: {len(actions)} steps, goal_reward: {total_goal_reward:.1f}, avg_dist_change: {avg_distance_change:.2f}")
        
        # Update policy with collected data
        policy_losses = self.update_policy_with_diagnostics(states, actions, rewards, values, log_probs)
        
        return states, actions, rewards, goal_reached
        
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy network.
        
        Args:
            state: Agent position [batch_size, 2] or [2]
            goal: Goal position [batch_size, 2] or [2]
            
        Returns:
            action_logits: Raw action logits [batch_size, 7]
            value: State value [batch_size, 1]
        """
        # Handle single sample input
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
            
        # Combine state and goal
        #combined = torch.cat([state, goal], dim=-1)  # [batch_size, 4]
        
        combined = torch.cat([state, goal], dim=-1).unsqueeze(1)  # [batch_size, 1, 4] for LSTM
        
        lstm_out, self.hidden_state = self.lstm(combined, self.hidden_state) 
        # Process through network
        #features = self.net(combined)               # [batch_size, 64]
        features = lstm_out.squeeze(1)              # [batch_size, 64]
        action_logits = self.actor(features)        # [batch_size, 7]
        value = self.critic(features)               # [batch_size, 1]
        
        return action_logits, value
    
    def get_action(self, state: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy for a single state-goal pair.
        Now with action masking for navigation-only actions - FIXED tensor indexing.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        goal_tensor = torch.tensor(goal, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            action_logits, value = self.forward(state_tensor, goal_tensor)
            
        # MASK NON-NAVIGATION ACTIONS
        # Allow only: turn_left (0), turn_right (1), move_forward (2)
        navigation_mask = torch.tensor([0, 1, 2], device=self.device)
        
        # FIX: Handle batch dimension properly
        if action_logits.dim() > 1:
            action_logits = action_logits.squeeze(0)  # Remove batch dimension: [1, 7] -> [7]
        
        masked_logits = torch.full_like(action_logits, float('-inf'))
        masked_logits[navigation_mask] = action_logits[navigation_mask]
        
        # Sample action from masked distribution
        action_probs = F.softmax(masked_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
    def update_policy(self, states: List, actions: List[int], rewards: List[float], 
                         values: List[torch.Tensor], log_probs: List[torch.Tensor]):
            """
            Update policy parameters using A2C algorithm.
            
            Args:
                states: List of (state, goal) tuples
                actions: List of action indices  
                rewards: List of rewards
                values: List of value estimates
                log_probs: List of action log probabilities
            """
            if len(rewards) == 0:
                return
                
            # Convert to tensors
            returns = []
            discounted_reward = 0
            
            # Compute discounted returns (backwards)
            for reward in reversed(rewards):
                discounted_reward = reward + self.gamma * discounted_reward
                returns.insert(0, discounted_reward)
                
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            values = torch.stack(values).squeeze()  # Ensure consistent shape
            log_probs = torch.stack(log_probs)
            actions = torch.tensor(actions, dtype=torch.long, device=self.device)
            
            # Ensure all tensors have same shape
            if values.dim() == 0:  # Single value case
                values = values.unsqueeze(0)
            if returns.dim() == 0:
                returns = returns.unsqueeze(0)
            
            # Compute advantages
            advantages = returns - values
            
            # Normalize advantages (handle single-step episodes)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages  # Don't normalize single values
            
            # Compute losses
            policy_loss = -(log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, returns)
            
            # Entropy for exploration (recompute from states)
            entropy_loss = 0
            for (state, goal) in states:
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
                goal_tensor = torch.tensor(goal, dtype=torch.float32, device=self.device)
                action_logits, _ = self.forward(state_tensor, goal_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                entropy_loss += -(action_probs * torch.log(action_probs + 1e-8)).sum()
            
            entropy_loss = entropy_loss / len(states)
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_coef * value_loss - 
                         self.entropy_coef * entropy_loss)
            
            # Update parameters
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
    def update_policy_with_diagnostics(self, states: List, actions: List[int], rewards: List[float], 
                     values: List[torch.Tensor], log_probs: List[torch.Tensor]):
        """
        Update policy parameters using A2C algorithm with detailed diagnostics.
        """
        if len(rewards) == 0:
            return {}
            
        # Convert to tensors
        returns = []
        discounted_reward = 0
        
        # Compute discounted returns (backwards)
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        
        # Ensure all tensors have same shape
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        
        # Compute advantages
        advantages = returns - values
        
        # DIAGNOSTICS: Track advantage statistics
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item() if len(advantages) > 1 else 0.0
        
        # Normalize advantages (handle single-step episodes)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store pre-update parameters for gradient analysis
        pre_update_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Compute losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        
        # Entropy for exploration
        entropy_loss = 0
        for (state, goal) in states:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            goal_tensor = torch.tensor(goal, dtype=torch.float32, device=self.device)
            action_logits, _ = self.forward(state_tensor, goal_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            entropy_loss += -(action_probs * torch.log(action_probs + 1e-8)).sum()
        
        entropy_loss = entropy_loss / len(states)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss - 
                     self.entropy_coef * entropy_loss)
        
        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # DIAGNOSTICS: Compute gradient norms before clipping
        total_grad_norm = 0
        param_count = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2)
                total_grad_norm += param_grad_norm.item() ** 2
                param_count += 1
        total_grad_norm = total_grad_norm ** (1. / 2)
        
        self.optimizer.step()
        
        # DIAGNOSTICS: Compute parameter change magnitude
        param_change_norm = 0
        for name, param in self.named_parameters():
            if name in pre_update_params:
                change = param - pre_update_params[name]
                param_change_norm += change.norm().item() ** 2
        param_change_norm = param_change_norm ** (1. / 2)
        
        # Return diagnostic information
        diagnostics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'advantage_mean': advantage_mean,
            'advantage_std': advantage_std,
            'grad_norm': total_grad_norm,
            'param_change_norm': param_change_norm,
            'returns_mean': returns.mean().item(),
            'returns_std': returns.std().item(),
            'values_mean': values.mean().item()
        }
        
        # Print diagnostics every few episodes (could be controlled with a flag)
        if len(states) > 5:  # Only for longer episodes
            if old_diag_phase_1:
                print(f"    Policy update: loss={total_loss:.4f}, adv_mean={advantage_mean:.3f}, grad_norm={total_grad_norm:.4f}, param_change={param_change_norm:.6f}")
        
        return diagnostics
        
    def collect_episodes_from_position(self, env, start_pos: Tuple[int, int], 
                                     num_episodes: int = 6, 
                                     max_episode_length: int = 50,  # ADD THIS
                                     vae_system=None,
                                     curiosity_weight: float = 0.5,
                                     use_curiosity: bool = True) -> List:
        """
        Collect multiple episodes starting from a specific position (pivotal state).
        Used in alternating training to explore from discovered pivotal states.
        
        Args:
            env: Environment
            start_pos: Starting position (pivotal state)
            num_episodes: Number of episodes to collect from this position
            vae_system: VAE for curiosity rewards
            curiosity_weight: Weight for curiosity rewards
            
        Returns:
            List of episode data for adding to buffer
        """
        episodes_data = []
        
        for episode in range(num_episodes):
            try:
                states, actions, rewards, goal_reached = self.train_goal_policy_episode(
                    env, start_pos,
                    max_episode_length=max_episode_length,  # USE IT
                    vae_system=vae_system,
                    curiosity_weight=curiosity_weight
                )
                
                # Store episode data in format compatible with StatBuffer
                episode_data = {
                    'states': [state for state, goal in states],  # Extract just positions
                    'actions': actions,
                    'rewards': rewards,
                    'goal_reached': goal_reached,
                    'start_pos': start_pos
                }
                episodes_data.append(episode_data)
                
            except Exception as e:
                print(f"    Episode {episode} from {start_pos} failed: {e}")
                continue
        
        return episodes_data
    
    def discover_edges_between_pivotal_states(self, env, pivotal_states: List[Tuple[int, int]], 
                                            max_walk_length: int = 20,
                                            num_attempts: int = 50) -> Dict[Tuple, List[Tuple]]:
        """
        Discover edges between pivotal states using random walks.
        FIXED: Added boundary validation to prevent grid overflow.
        """
        discovered_edges = {}
        pivotal_set = set(pivotal_states)
        
        print(f"Discovering edges between {len(pivotal_states)} pivotal states...")
        
        for start_state in pivotal_states:
            if self.verbose:
                print(f"  Random walks from {start_state}:")
            
            for attempt in range(num_attempts):
                try:
                    # Reset environment and place agent at start state
                    env.reset()
                    env.agent_pos = start_state
                    env.agent_dir = 0  # Face right initially
                    current_pos = start_state
                    path = [start_state]
                    
                    # Perform random walk
                    for step in range(max_walk_length):
                        # Random action selection (only navigation actions)
                        action = random.choice([0, 1, 2])  # turn_left, turn_right, move_forward
                        
                        # BOUNDARY CHECK: Predict next position before taking step
                        if action == 2:  # move_forward
                            # Get agent's current direction
                            direction = env.agent_dir
                            
                            # Calculate forward position based on direction
                            if direction == 0:    # Right
                                next_pos = (current_pos[0] + 1, current_pos[1])
                            elif direction == 1:  # Down  
                                next_pos = (current_pos[0], current_pos[1] + 1)
                            elif direction == 2:  # Left
                                next_pos = (current_pos[0] - 1, current_pos[1])
                            elif direction == 3:  # Up
                                next_pos = (current_pos[0], current_pos[1] - 1)
                            else:
                                next_pos = current_pos
                            
                            # Check if next position is within bounds
                            if not (1 <= next_pos[0] <= env.size-2 and 1 <= next_pos[1] <= env.size-2):
                                # Skip this action - would go out of bounds
                                continue
                        
                        # Take step (now safe from boundary violations)
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        # Get new position
                        if hasattr(env, 'agent_pos') and env.agent_pos is not None:
                            new_pos = tuple(env.agent_pos)
                        else:
                            new_pos = current_pos
                        
                        if new_pos != current_pos:
                            path.append(new_pos)
                            current_pos = new_pos
                            
                            # Check if reached another pivotal state
                            if new_pos in pivotal_set and new_pos != start_state:
                                # Check if path intersects other pivotal states (except start/end)
                                intersected_pivotal = False
                                for intermediate_pos in path[1:-1]:  # Exclude start and end
                                    if intermediate_pos in pivotal_set:
                                        intersected_pivotal = True
                                        break
                                
                                if not intersected_pivotal:
                                    # Valid edge found
                                    edge_key = (start_state, new_pos)
                                    if edge_key not in discovered_edges:
                                        discovered_edges[edge_key] = path.copy()
                                        if self.verbose:
                                            print(f"    Found edge {start_state} -> {new_pos} (length: {len(path)})")
                                break
                        
                        if terminated or truncated:
                            break
                            
                except Exception as e:
                    print(f"    Random walk attempt {attempt} from {start_state} failed: {e}")
                    continue
        
        print(f"Discovered {len(discovered_edges)} edges between pivotal states")
        return discovered_edges
    
    def refine_paths_with_goal_policy(self, env, raw_edges: Dict[Tuple, List[Tuple]]) -> Dict[Tuple, Tuple[List[Tuple], int]]:
        """
        Refine discovered edge paths using goal-conditioned policy.
        
        Returns:
            Dict of (start, end) -> (refined_path, weight)
        """
        refined_edges = {}

        
        print("Refining edge paths with goal-conditioned policy...")
        
        for (start_state, end_state), raw_path in raw_edges.items():
            print(f"  Refining {start_state} -> {end_state}")
            
            try:
                env.reset()
                env.agent_pos = start_state
                env.agent_dir = 0
                current_pos = start_state
                
                refined_path = [start_state]
                max_refinement_steps = len(raw_path) + 5
                
                for step in range(max_refinement_steps):
                    action, log_prob, value = self.get_action(current_pos, end_state)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if hasattr(env, 'agent_pos') and env.agent_pos is not None:
                        new_pos = tuple(env.agent_pos)
                    else:
                        new_pos = current_pos
                    
                    if new_pos != current_pos:
                        refined_path.append(new_pos)
                        current_pos = new_pos
                    
                    if current_pos == end_state:
                        if len(refined_path) <= len(raw_path) * 1.2:
                            edge_weight = len(refined_path) - 1
                            refined_edges[(start_state, end_state)] = (refined_path, edge_weight)
                            print(f"    Refined: {len(refined_path)} nodes")
                        else:
                            edge_weight = len(raw_path) - 1
                            refined_edges[(start_state, end_state)] = (raw_path, edge_weight)
                            print(f"    Kept raw: {len(raw_path)} nodes (refinement too long)")
                        break
                        
                    if terminated or truncated:
                        break
                else:
                    edge_weight = len(raw_path) - 1
                    refined_edges[(start_state, end_state)] = (raw_path, edge_weight)
                    print(f"    Policy failed, kept raw: {len(raw_path)} nodes")
                    
            except Exception as e:
                edge_weight = len(raw_path) - 1
                refined_edges[(start_state, end_state)] = (raw_path, edge_weight)
                print(f"    Error during refinement, kept raw: {len(raw_path)} nodes - {e}")
        
        return refined_edges
    
    def construct_world_graph(self, pivotal_states: List[Tuple[int, int]], 
                        refined_edges: Dict[Tuple, Tuple[List[Tuple], int]]) -> GraphManager:
        """
        Construct the final world graph with nodes and weighted edges.
        
        Args:
            refined_edges: Dict of (start, end) -> (path, weight)
        """
        world_graph = GraphManager()
        
        print("Constructing world graph...")
        
        # Add all pivotal states as nodes
        for state in pivotal_states:
            world_graph.add_node(state)
        
        # Add refined edges with weights only (no action sequences)
        for (start_state, end_state), (path, weight) in refined_edges.items():
            world_graph.add_edge(start_state, end_state, weight)
            if self.verbose:
                print(f"  Edge: {start_state} -> {end_state}, weight: {weight}")
        
        print(f"World graph constructed: {len(pivotal_states)} nodes, {len(refined_edges)} edges")
        
        return world_graph

    def complete_world_graph_discovery(self, env, pivotal_states: List[Tuple[int, int]]) -> GraphManager:
        """
        Complete Phase 1 by discovering edges and constructing world graph.
        
        Args:
            env: Environment
            pivotal_states: Discovered pivotal states from VAE
            
        Returns:
            GraphManager: Complete world graph
        """
        print("\n" + "="*60)
        print("COMPLETING WORLD GRAPH DISCOVERY (Phase 1)")
        print("="*60)
        
        # Step 1: Discover edges through random walks
        raw_edges = self.discover_edges_between_pivotal_states(env, pivotal_states)
        
        # Step 2: Refine paths using goal-conditioned policy
        refined_edges = self.refine_paths_with_goal_policy(env, raw_edges)
        
        # Step 3: Construct final world graph
        world_graph = self.construct_world_graph(pivotal_states, refined_edges)
        
        # Step 4: Validate graph connectivity
        print(f"\nWorld Graph Summary:")
        print(f"  Nodes (pivotal states): {len(pivotal_states)}")
        print(f"  Edges: {len(refined_edges)}")
        
        # Show sample shortest paths
        print(f"  Sample shortest paths:")
        for i, start_state in enumerate(pivotal_states[:3]):
            for j, end_state in enumerate(pivotal_states[:3]):
                if i != j:
                    path, distance = world_graph.shortest_path(start_state, end_state)
                    if path:
                        print(f"    {start_state} -> {end_state}: {distance} steps")
        
        print("="*60)
        print("PHASE 1 COMPLETE: World graph discovery finished")
        print("="*60)
        
        return world_graph
    
#c