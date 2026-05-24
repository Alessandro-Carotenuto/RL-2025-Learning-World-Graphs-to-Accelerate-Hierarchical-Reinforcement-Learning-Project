import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from typing import List, Tuple, Dict

# PROJECT-SPECIFIC IMPORTS

from utils.misc import manhattan_distance,sample_goal_position
from utils.graph_manager import GraphManager

#-----------------------------------------------------------------------------

class GoalConditionedPolicy(nn.Module):
    """
    GOAL-CONDITIONED POLICY FOR WORLD GRAPH DISCOVERY
    """
    
    def __init__(self, lr: float = 5e-3, verbose: bool = False, maze_size: int = 24, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.verbose = verbose
        self.device = device
        self.maze_size = maze_size
        
        # NETWORK COMPONENTS (6 inputs: state_x, state_y, dir_sin, dir_cos, goal_dx, goal_dy)
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        ).to(device)

        self.actor = nn.Linear(64, 7).to(device)
        self.critic = nn.Linear(64, 1).to(device)

        # OPTIMIZER
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # HYPERPARAMETERS
        self.gamma = 0.99
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        
    def train_goal_policy_episode(self, env, start_pos: Tuple[int, int], 
                        max_episode_length: int = 50,
                        vae_system=None, 
                        curiosity_weight: float = 1.0) -> Tuple[List, List, List, bool]:
        """
        TRAIN POLICY FOR ONE EPISODE (GOAL-CONDITIONED)
        """
        # SAMPLE GOAL AND RESET STATE
        goal_pos = sample_goal_position(env, start_pos, max_distance=20)

        # INITIALIZE EPISODE BUFFERS
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        # TRACK VISITED STATES FOR CURIOSITY
        visited_states = [start_pos]

        # RESET ENVIRONMENT POSITION
        obs = env.reset()
        env.agent_pos = start_pos
        env.agent_dir = 0
        current_pos = start_pos
        current_dir = env.agent_dir
        goal_reached = False

        episode_trajectory = []
        # RUN EPISODE
        for step in range(max_episode_length):
            # Get action from policy
            action, log_prob, value = self.get_action(current_pos, current_dir, goal_pos)
            episode_trajectory.append((current_pos, action))
            # STORE STATE-ACTION
            states.append((current_pos, current_dir, goal_pos))
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)

            # STEP ENVIRONMENT
            obs, env_reward, terminated, truncated, info = env.step(action)

            # Get new position
            if hasattr(env, 'agent_pos') and env.agent_pos is not None:
                next_pos = tuple(env.agent_pos)
            else:
                next_pos = current_pos

            # UPDATE VISITED STATES
            if next_pos != current_pos:
                visited_states.append(next_pos)

            old_distance = manhattan_distance(current_pos, goal_pos)
            new_distance = manhattan_distance(next_pos, goal_pos)
            goal_reward = 10.0 if next_pos == goal_pos else 0.0
            progress_reward = 0.2 if new_distance < old_distance else 0.0
            step_penalty = -0.01

            # CURIOSITY REWARD COMPUTATION
            curiosity_reward = 0.0
            if vae_system is not None and len(visited_states) > 1:
                window_size = min(5, len(episode_trajectory))
                recent_trajectory = episode_trajectory[-window_size:]
                base_curiosity = vae_system.compute_curiosity_reward_from_trajectory(recent_trajectory)

                curiosity_reward = base_curiosity * curiosity_weight
                curiosity_reward = min(1.0, curiosity_reward)

            total_reward = goal_reward + progress_reward + step_penalty + curiosity_reward
            rewards.append(total_reward)

            # TERMINATION CHECKS
            if goal_reward > 0:
                goal_reached = True
                break

            if terminated or truncated:
                break

            current_pos = next_pos
            current_dir = env.agent_dir

        # UPDATE POLICY
        self.update_policy(states, actions, rewards, values, log_probs)

        return states, actions, rewards, goal_reached
        
    def forward(self, state: torch.Tensor, agent_dir: int, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the policy network.

        Args:
            state: Agent position [batch_size, 2] or [2]
            agent_dir: Current facing direction (0-3)
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

        batch_size = state.shape[0]
        # dir as (sin, cos) — circular encoding, adjacent dirs always distance √2 apart.
        # Scalar dir/3 makes dir=3 (1.0) and dir=0 (0.0) appear maximally different even though
        # they are 1 turn apart, causing gradient conflicts in learning turning behavior.
        state_norm = state / self.maze_size
        goal_rel   = (goal - state) / self.maze_size
        dir_sin = torch.full((batch_size, 1), math.sin(agent_dir * math.pi / 2), dtype=torch.float32, device=self.device)
        dir_cos = torch.full((batch_size, 1), math.cos(agent_dir * math.pi / 2), dtype=torch.float32, device=self.device)

        combined = torch.cat([state_norm, dir_sin, dir_cos, goal_rel], dim=-1)  # [batch_size, 6]
        features = self.net(combined)                        # [batch_size, 64]
        action_logits = self.actor(features)        # [batch_size, 7]
        value = self.critic(features)               # [batch_size, 1]
        
        return action_logits, value
    
    def get_action(self, state: Tuple[int, int], agent_dir: int, goal: Tuple[int, int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy for a single state-goal pair.
        Now with action masking for navigation-only actions - FIXED tensor indexing.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        goal_tensor = torch.tensor(goal, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_logits, value = self.forward(state_tensor, agent_dir, goal_tensor)
            
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
                      values: List[torch.Tensor], log_probs: List[torch.Tensor],
                      diagnostics: bool = False):
        if len(rewards) == 0:
            return {} if diagnostics else None

        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)

        if values.dim() == 0:
            values = values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)

        advantages = returns - values

        if diagnostics:
            advantage_mean = advantages.mean().item()
            advantage_std = advantages.std().item() if len(advantages) > 1 else 0.0
            pre_update_params = {name: param.clone() for name, param in self.named_parameters()}

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)

        entropy_loss = 0
        for (state, agent_dir, goal) in states:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            goal_tensor = torch.tensor(goal, dtype=torch.float32, device=self.device)
            action_logits, _ = self.forward(state_tensor, agent_dir, goal_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            entropy_loss += -(action_probs * torch.log(action_probs + 1e-8)).sum()
        entropy_loss = entropy_loss / len(states)

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        if diagnostics:
            total_grad_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in self.parameters() if p.grad is not None
            ) ** 0.5

        self.optimizer.step()

        if not diagnostics:
            return None

        param_change_norm = sum(
            (param - pre_update_params[name]).norm().item() ** 2
            for name, param in self.named_parameters()
            if name in pre_update_params
        ) ** 0.5

        return {
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
            'values_mean': values.mean().item(),
        }
        
    def collect_episodes_from_position(self, env, start_pos: Tuple[int, int], 
                                     num_episodes: int = 6, 
                                     max_episode_length: int = 50,
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
                    'states': [state for state, agent_dir, goal in states],  # Extract just positions
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
                                            num_attempts: int = 70) -> Dict[Tuple, List[Tuple]]:
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
                max_refinement_steps = len(raw_path) * 5 + 20
                
                for step in range(max_refinement_steps):
                    action, log_prob, value = self.get_action(current_pos, env.agent_dir, end_state)
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
        """
        world_graph = GraphManager()
        
        print("Constructing world graph...")
        
        # Add all pivotal states as nodes
        for state in pivotal_states:
            world_graph.add_node(state)
        
        # Add refined edges with weights AND the full path
        for (start_state, end_state), (path, weight) in refined_edges.items():
            # *** MODIFICATION HERE ***
            world_graph.add_edge(start_state, end_state, weight, path)
            # *** END MODIFICATION ***
            if self.verbose:
                print(f"  Edge: {start_state} -> {end_state}, weight: {weight}, path_len: {len(path)}")
        
        print(f"World graph constructed: {len(pivotal_states)} nodes, {len(refined_edges)} edges")
        
        return world_graph
   
    def complete_world_graph_discovery(self, env, pivotal_states: List[Tuple[int, int]],
                                       graph_walk_length: int = 20,
                                       graph_num_attempts: int = 70) -> GraphManager:
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
        raw_edges = self.discover_edges_between_pivotal_states(env, pivotal_states,
                                                                max_walk_length=graph_walk_length,
                                                                num_attempts=graph_num_attempts)
        
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
