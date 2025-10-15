import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import random
import numpy as np
from utils.optimal_reward_computer import compute_optimal_reward_for_episode
from collections import Counter
from utils.misc import manhattan_distance

diag=False
diag2=False
diag3=False

class HierarchicalManager(nn.Module):
    """
    Phase 2: Hierarchical Manager implementing Wide-then-Narrow goal selection.
    Paper: "Manager uses two-step 'Wide-then-Narrow' goal descriptions"
    """
    
    def __init__(self, 
                 pivotal_states: List[Tuple[int, int]],
                 neighborhood_size: int = 3,
                 lr: float = 5e-3,
                 horizon: int = 15,
                 diagnostic_interval: int = 30,  # NEW
                 diagnostic_checkstart: bool = True,  # NEW
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            pivotal_states: List of discovered pivotal state coordinates
            neighborhood_size: N×N local area around wide goal (paper uses N×N)
            lr: Learning rate
            horizon: Manager operates at c-step horizon
            device: Computing device
        """
        super().__init__()
        
        self.device = device
        self.pivotal_states = pivotal_states
        self.neighborhood_size = neighborhood_size
        self.horizon = horizon
        
        # A2C-LSTM architecture (from paper: "Manager and Worker are both A2C-LSTMs")
        self.lstm = nn.LSTM(
            input_size=4,  # [state_x, state_y, prev_gw_x, prev_gw_y] 
            hidden_size=64,
            num_layers=1,
            batch_first=True
        ).to(device)
        
        # Wide policy: π^ω(gw,t|st) - categorical over pivotal states
        self.wide_head = nn.Linear(64, len(pivotal_states)).to(device)
        
        # Narrow policy: π^n(gn,t|st, gw,t, sw,t) - categorical over neighborhood
        self.narrow_head = nn.Linear(64 + 2, neighborhood_size**2).to(device)  # +2 for gw coords
        
        # Value function V(st)
        self.critic = nn.Linear(64, 1).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Manager state
        self.hidden_state = None
        self.prev_wide_goal = (0, 0)  # Previous wide goal for LSTM input
        
        # Hyperparameters
        self.gamma = 0.99
        self.entropy_coef = 1e-3    
        self.value_coef = 0.5

        # Diagnostic parameters
        self.diagnostic_interval = diagnostic_interval
        self.diagnostic_checkstart = diagnostic_checkstart
    
    def reset_manager_state(self):
        """Reset LSTM hidden state and previous goals."""
        self.hidden_state = None
        self.prev_wide_goal = (0, 0)
    
    def get_neighborhood(self, wide_goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get N×N neighborhood around wide goal.
        Paper: "zooms attention to an N × N local area sw,t around gw"
        """
        gw_x, gw_y = wide_goal
        neighborhood = []
        
        # Generate N×N grid centered on wide goal
        offset = self.neighborhood_size // 2
        for dx in range(-offset, offset + 1):
            for dy in range(-offset, offset + 1):
                neighbor = (gw_x + dx, gw_y + dy)
                neighborhood.append(neighbor)
        
        return neighborhood
    
    def forward(self, state: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: Wide goal selection only.
        
        Args:
            state: Current agent position (x, y)
            
        Returns:
            wide_logits: Logits over pivotal states [num_pivotal_states]
            features: LSTM features for narrow policy [64]
            value: State value estimate [1]
        """
        # Prepare LSTM input: [state_x, state_y, prev_gw_x, prev_gw_y]
        lstm_input = torch.tensor([
            state[0], state[1], 
            self.prev_wide_goal[0], self.prev_wide_goal[1]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        
        # LSTM forward pass
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        features = lstm_out.squeeze(0).squeeze(0)  # [64]
        
        # Wide policy: π^ω(gw,t|st)
        wide_logits = self.wide_head(features)  # [num_pivotal_states]
        
        # Value function
        value = self.critic(features)  # [1]
        
        return wide_logits, features, value
    
    def select_wide_goal(self, state: Tuple[int, int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select wide goal from pivotal states.
        Paper: π^ω(gw,t|st) outputs a "wide" subgoal gw ∈ V where V = Vp
        """
        wide_logits, features, value = self.forward(state)
        
        # Sample from categorical distribution
        wide_probs = F.softmax(wide_logits, dim=0)
        wide_dist = torch.distributions.Categorical(wide_probs)
        wide_idx = wide_dist.sample()
        wide_log_prob = wide_dist.log_prob(wide_idx)
        
        # Get actual pivotal state coordinates
        wide_goal = self.pivotal_states[wide_idx.item()]
        
        # Update previous wide goal for next LSTM input
        self.prev_wide_goal = wide_goal
        
        return wide_idx.item(), wide_log_prob, value.squeeze()
    
    def select_narrow_goal(self, state: Tuple[int, int], wide_goal: Tuple[int, int]) -> Tuple[Tuple[int, int], torch.Tensor]:
        """
        Select narrow goal from neighborhood around wide goal.
        Paper: π^n(gn,t|st, gw,t, sw,t) selects final "narrow" goal gn ∈ sw,t
        """
        # Get LSTM features (recompute to ensure consistency)
        _, features, _ = self.forward(state)
        
        # Concatenate features with wide goal coordinates
        wide_goal_tensor = torch.tensor(wide_goal, dtype=torch.float32, device=self.device)
        narrow_input = torch.cat([features, wide_goal_tensor])  # [64 + 2]
        
        # Narrow policy: π^n(gn,t|st, gw,t, sw,t)
        narrow_logits = self.narrow_head(narrow_input)  # [neighborhood_size^2]
        
        # Sample from categorical distribution
        narrow_probs = F.softmax(narrow_logits, dim=0)
        narrow_dist = torch.distributions.Categorical(narrow_probs)
        narrow_idx = narrow_dist.sample()
        narrow_log_prob = narrow_dist.log_prob(narrow_idx)
        
        # Convert index to actual coordinates
        neighborhood = self.get_neighborhood(wide_goal)
        narrow_goal = neighborhood[narrow_idx.item()]
        
        return narrow_goal, narrow_log_prob
    
    def get_manager_action(self, state: Tuple[int, int], step_count: int = 0):
        """Complete Wide-then-Narrow goal selection with hidden state diagnostics."""
        
        # Configurable diagnostic printing
        if self.diagnostic_checkstart and step_count < 15:
            verbose = True
        elif step_count % self.diagnostic_interval == 0:
            verbose = True
        else:
            verbose = False
        
        if verbose:
            print(f"\n[Manager Action] Step {step_count}")
            print(f"  Current state: {state}")
            print(f"  Hidden state exists: {self.hidden_state is not None}")
            if self.hidden_state is not None:
                h, c = self.hidden_state
                print(f"  Hidden state norms: h={h.norm().item():.3f}, c={c.norm().item():.3f}")
        
        # Step 1: Wide goal selection
        wide_idx, wide_log_prob, value = self.select_wide_goal(state)
        wide_goal = self.pivotal_states[wide_idx]
        
        # Get distribution for diagnostics
        if verbose:
            with torch.no_grad():
                wide_logits, _, _ = self.forward(state)
                wide_probs = F.softmax(wide_logits, dim=0)
                
                # Top 5 logits and probs
                top_k = min(5, len(wide_logits))
                top_logits, top_indices = wide_logits.topk(top_k)
                top_probs = wide_probs[top_indices]
                
                print(f"  Wide goal selection:")
                print(f"    Top {top_k} logits: {top_logits.tolist()}")
                print(f"    Top {top_k} probs: {top_probs.tolist()}")
                print(f"    Selected idx: {wide_idx}, goal: {wide_goal}")
                
                # Entropy
                entropy = -(wide_probs * torch.log(wide_probs + 1e-8)).sum()
                max_entropy = torch.log(torch.tensor(float(len(self.pivotal_states))))
                print(f"    Entropy: {entropy.item():.3f} / {max_entropy.item():.3f} ({100*entropy/max_entropy:.1f}%)")
        
        # Step 2: Narrow goal selection  
        narrow_goal, narrow_log_prob = self.select_narrow_goal(state, wide_goal)
        
        # Combined log probability
        combined_log_prob = wide_log_prob + narrow_log_prob
        
        if verbose:
            print(f"  Narrow goal: {narrow_goal}")
            print(f"  Combined log prob: {combined_log_prob.item():.3f}")
            print(f"  Value estimate: {value.item():.3f}")
        
        return wide_goal, narrow_goal, combined_log_prob, value
        
    # Transfer learning initialization (broken? just copying heads i guess)
    # def initialize_from_goal_policy(self, goal_policy):
    #     """
    #     Transfer learning: Initialize Manager with better scaling.
    #     """
    #     with torch.no_grad():
    #         # Use standard initialization (gain=1.0, not 0.1)
    #         for name, param in self.lstm.named_parameters():
    #             if 'weight_ih' in name:
    #                 nn.init.xavier_uniform_(param, gain=1.0)  # ← FIX: gain=1.0
    #             elif 'weight_hh' in name:
    #                 nn.init.orthogonal_(param, gain=1.0)  # ← Better for recurrent
    #             elif 'bias' in name:
    #                 nn.init.zeros_(param)
            
    #         # Initialize output heads with standard gain
    #         nn.init.xavier_uniform_(self.wide_head.weight, gain=1.0)
    #         nn.init.zeros_(self.wide_head.bias)
    #         nn.init.xavier_uniform_(self.narrow_head.weight, gain=1.0)
    #         nn.init.zeros_(self.narrow_head.bias)
            
    #         # Initialize critic
    #         if hasattr(goal_policy, 'critic'):
    #             self.critic.weight.copy_(goal_policy.critic.weight)
    #             self.critic.bias.copy_(goal_policy.critic.bias)
    #         else:
    #             nn.init.xavier_uniform_(self.critic.weight, gain=1.0)
    #             nn.init.zeros_(self.critic.bias) 
        
    #     print(f"Manager initialized with standard scaling (gain=1.0)")

    # Actual initialization from goal policy (LSTM copying)
    def initialize_from_goal_policy(self, goal_policy):
        with torch.no_grad():
            # Similar LSTM copying logic as Worker
            if hasattr(goal_policy, 'lstm'):
                for name, param in goal_policy.lstm.named_parameters():
                    if name in dict(self.lstm.named_parameters()):
                        manager_param = dict(self.lstm.named_parameters())[name]
                        if param.shape == manager_param.shape:
                            manager_param.copy_(param)
            
            # Copy critic only (manager has different heads)
            if hasattr(goal_policy, 'critic'):
                self.critic.weight.copy_(goal_policy.critic.weight)
                self.critic.bias.copy_(goal_policy.critic.bias)
            
            # Initialize manager heads with small weights
            nn.init.xavier_uniform_(self.wide_head.weight, gain=0.5)
            nn.init.zeros_(self.wide_head.bias)
            nn.init.xavier_uniform_(self.narrow_head.weight, gain=0.5)
            nn.init.zeros_(self.narrow_head.bias)
        
        print("Manager initialized from goal policy LSTM")


    def update_policy(self, states, wide_goals, narrow_goals, rewards, values, log_probs, entropies, step_count=0):
        """Update Manager policy with wide + narrow entropy regularization."""
        if len(rewards) == 0:
            return
        
        # Diagnostic printing
        if self.diagnostic_checkstart and step_count < 15:
            verbose = True
        elif step_count % self.diagnostic_interval == 0:
            verbose = True
        else:
            verbose = False
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"[Manager Update Debug] Step {step_count}")
            print(f"{'='*70}")
        
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values_tensor = torch.stack(values).squeeze()
        log_probs_tensor = torch.stack(log_probs)
        wide_entropies_tensor = torch.stack(entropies)  # These are wide entropies from get_manager_action
        
        # Fix dimensions
        if values_tensor.dim() == 0:
            values_tensor = values_tensor.unsqueeze(0)
        if rewards_tensor.dim() == 0:
            rewards_tensor = rewards_tensor.unsqueeze(0)
        
        if verbose:
            print(f"Batch size: {len(rewards)}")
            print(f"Rewards: {rewards_tensor.tolist()}")
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # GAE
        gae_lambda = 0.95
        advantages = torch.zeros_like(rewards_tensor)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards_tensor[t] - values_tensor[t]
            else:
                delta = rewards_tensor[t] + self.gamma * values_tensor[t + 1] - values_tensor[t]
            gae = delta + self.gamma * gae_lambda * gae
            advantages[t] = gae
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy and value losses
        policy_loss = -(advantages.detach() * log_probs_tensor).mean()
        value_loss = F.mse_loss(values_tensor, returns)
        
        # Compute narrow entropy: E_gw[H(π^n|gw)]
        narrow_entropies = []
        for i, (state, wide_goal) in enumerate(zip(states, wide_goals)):
            # Recompute narrow distribution for this state and wide goal
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            wide_logits, features, _ = self.forward(state)
            
            # Get narrow logits
            wide_goal_tensor = torch.tensor(wide_goal, dtype=torch.float32, device=self.device)
            narrow_input = torch.cat([features, wide_goal_tensor])
            narrow_logits = self.narrow_head(narrow_input)
            
            # Compute entropy
            narrow_probs = F.softmax(narrow_logits, dim=0)
            narrow_entropy = -(narrow_probs * torch.log(narrow_probs + 1e-8)).sum()
            narrow_entropies.append(narrow_entropy)
        
        narrow_entropy_mean = torch.stack(narrow_entropies).mean()
        wide_entropy_mean = wide_entropies_tensor.mean()
        
        if verbose:
            print(f"\nLoss components:")
            print(f"  Policy loss: {policy_loss.item():.6f}")
            print(f"  Value loss: {value_loss.item():.6f}")
            print(f"  Wide entropy: {wide_entropy_mean.item():.6f}")
            print(f"  Narrow entropy: {narrow_entropy_mean.item():.6f}")
            print(f"  Entropy coef: {self.entropy_coef}")
        
        # Combined loss with both entropies (paper: H(π^ω) + H(π^n|gw))
        total_loss = (policy_loss + 
                    self.value_coef * value_loss - 
                    self.entropy_coef * (wide_entropy_mean + narrow_entropy_mean))
        
        if verbose:
            print(f"  Total loss: {total_loss.item():.6f}")
        
        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        if verbose:
            print(f"{'='*70}\n")

class HierarchicalWorker(nn.Module):
    """
    Phase 2: Hierarchical Worker that executes Manager's goals.
    Paper: "Worker can leverage the graph to easily traverse to pivotal states"
    """
    
    def __init__(self,
                 world_graph,
                 pivotal_states: List[Tuple[int, int]],
                 lr: float = 5e-3,
                 verbose: bool =False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            world_graph: GraphManager with edges between pivotal states
            pivotal_states: List of pivotal state coordinates
            lr: Learning rate
            device: Computing device
        """
        self.verbose = verbose
        super().__init__()
        
        self.device = device
        self.world_graph = world_graph
        self.pivotal_states = set(pivotal_states)
        
        # A2C-LSTM architecture
        self.lstm = nn.LSTM(
            input_size=6,  # [state_x, state_y, gw_x, gw_y, gn_x, gn_y]
            hidden_size=64,
            num_layers=1,
            batch_first=True
        ).to(device)
        
        self.actor = nn.Linear(64, 3).to(device)
        self.critic = nn.Linear(64, 1).to(device)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Worker state
        self.hidden_state = None
        self.current_traversal_path = []
        self.traversal_step = 0
        self.current_edge_actions = None
        self.current_action_idx = 0

        # --- NEW WORKER STATE VARIABLES ---
        self.current_traversal_path = []  # Path of PIVOTAL states, e.g., [(A), (B), (C)]
        self.traversal_step = 0           # Index into self.current_traversal_path

        self.current_edge_actions = []    # Action sequence for ONE edge, e.g., [1, 1, 2, 2]
        self.current_action_idx = 0       # Index into self.current_edge_actions
        
        # Hyperparameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
    
    def reset_worker_state(self):
        """Reset LSTM hidden state and traversal state."""
        self.hidden_state = None
        self.current_traversal_path = []
        self.traversal_step = 0
        self.current_edge_actions = None
        self.current_action_idx = 0
    
    def is_at_pivotal_state(self, state: Tuple[int, int]) -> bool:
        """Check if current state is a pivotal state."""
        return state in self.pivotal_states
    
    def plan_traversal(self, current_state: Tuple[int, int], target_state: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan graph traversal from current pivotal state to target pivotal state.
        Paper: "estimate optimal traversal route based on edge weights using dynamic programming"
        """
        if not self.is_at_pivotal_state(current_state):
            return None
        
        if not self.is_at_pivotal_state(target_state):
            return None
        
        # Use GraphManager's shortest path (Dijkstra)
        path, distance = self.world_graph.shortest_path(current_state, target_state)
        
        if path and len(path) > 1:
            return path
        
        return None
    
    def should_traverse(self, current_state: Tuple[int, int], wide_goal: Tuple[int, int]) -> bool:
        """
        Determine if Worker should initiate graph traversal.
        Paper: "Worker can traverse via world graph if it encounters pivotal state g'w with feasible connection to gw"
        """
        # Check if at pivotal state
        if not self.is_at_pivotal_state(current_state):
            return False
        
        # Check if not already at target
        if current_state == wide_goal:
            return False
        
        # Check if path exists in graph
        path = self.plan_traversal(current_state, wide_goal)
        
        return path is not None
    
    def forward(self, state: Tuple[int, int], wide_goal: Tuple[int, int], narrow_goal: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Worker network.
        
        Args:
            state: Current agent position
            wide_goal: Manager's wide goal
            narrow_goal: Manager's narrow goal
            
        Returns:
            action_logits: Logits over 3 navigation actions
            value: State value estimate
        """
        # LSTM input: [state_x, state_y, gw_x, gw_y, gn_x, gn_y]
        lstm_input = torch.tensor([
            state[0], state[1],
            wide_goal[0], wide_goal[1],
            narrow_goal[0], narrow_goal[1]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 6]
        
        # LSTM forward
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        features = lstm_out.squeeze(0).squeeze(0)  # [64]
        
        # Action and value
        action_logits = self.actor(features)  # [3]
        value = self.critic(features)  # [1]
        
        return action_logits, value
    
    def get_action(self, state: Tuple[int, int], wide_goal: Tuple[int, int], narrow_goal: Tuple[int, int], agent_dir: int):
        """Worker selects action by executing pre-computed paths or using policy."""

        # 1. CHECK IF WE SHOULD START A NEW TRAVERSAL
        # This happens only if we are NOT currently in a traversal.
        if not self.current_traversal_path and self.should_traverse(state, wide_goal):
            path = self.plan_traversal(state, wide_goal)
            if path:
                self.current_traversal_path = path
                self.traversal_step = 0
                if diag:
                    print(f"\n[WORKER DIAGNOSTIC] Initiating Traversal at {state}")
                    print(f"  - Target (gw): {wide_goal}")
                    print(f"  - Pivotal Path: {self.current_traversal_path}")
        
        # 2. EXECUTE THE CURRENT TRAVERSAL (if active)
        is_traversing = bool(self.current_traversal_path)

        if is_traversing:
            # Check if we need to load actions for a new edge segment
            if not self.current_edge_actions:
                if self.traversal_step < len(self.current_traversal_path) - 1:
                    start_node = self.current_traversal_path[self.traversal_step]
                    end_node = self.current_traversal_path[self.traversal_step + 1]

                    # CRITICAL SYNC CHECK before starting a new edge
                    if state != start_node:
                        if diag:
                            print(f"  - 🔴 DESYNC DETECTED! Agent at {state}, expected {start_node} to start edge.")
                            print(f"  - Aborting traversal.")
                        self.reset_worker_state()
                    else:
                        # Fetch the coordinate path and generate actions for it
                        coord_path = self.world_graph.get_edge_path(start_node, end_node)
                        if coord_path:
                            self.current_edge_actions = self.generate_actions_from_path(coord_path, agent_dir)
                            self.current_action_idx = 0
                            if diag:
                                print(f"  - Loading edge {start_node}->{end_node}. Generated {len(self.current_edge_actions)} actions.")
                        else:
                            # Path not found in graph, should not happen if plan is valid
                            if diag: print(f"  - 🔴 ERROR: Edge path for {start_node}->{end_node} not found!")
                            self.reset_worker_state()
                else:
                    # We have finished the last edge of the pivotal path
                    if diag: print("  - ✅ Traversal Complete. Switching to policy.")
                    self.reset_worker_state()

            # If we have actions to execute for the current edge, execute them
            if self.current_edge_actions and self.current_action_idx < len(self.current_edge_actions):
                action = self.current_edge_actions[self.current_action_idx]
                self.current_action_idx += 1

                # Check if this edge segment is now complete
                if self.current_action_idx >= len(self.current_edge_actions):
                    self.current_edge_actions = [] # Clear actions to load next edge
                    self.traversal_step += 1       # Move to next pivotal state in path
                    if diag:
                        print(f"  - Edge segment finished. Advancing to pivotal step {self.traversal_step}.")

                # Return the action from the pre-computed plan
                with torch.no_grad():
                    _, value = self.forward(state, wide_goal, narrow_goal)
                log_prob = torch.tensor(-1.0, device=self.device) # Dummy log_prob for planned actions
                return action, log_prob, value.squeeze()

        # 3. FALLBACK TO POLICY ACTION if not traversing
        with torch.no_grad():
            action_logits, value = self.forward(state, wide_goal, narrow_goal)

        action_probs = F.softmax(action_logits, dim=0)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob, value.squeeze()

    def _compute_required_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """
        Compute which direction agent must face to move from->to in one step.
        
        Returns:
            0: Right (+x)
            1: Down (+y)
            2: Left (-x)
            3: Up (-y)
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx > 0:
            return 0  # Right
        elif dy > 0:
            return 1  # Down
        elif dx < 0:
            return 2  # Left
        elif dy < 0:
            return 3  # Up
        else:
            # Same position (shouldn't happen in traversal)
            return 0

    def _compute_turn_action(self, current_dir: int, target_dir: int) -> int:
        """
        Compute ONE turn action to get closer to target direction.
        
        Args:
            current_dir: Current facing direction (0-3)
            target_dir: Target facing direction (0-3)
        
        Returns:
            0: turn_left
            1: turn_right
        """
        # Compute shortest rotation
        diff = (target_dir - current_dir) % 4
        
        if diff == 0:
            # Already facing target (shouldn't be called in this case)
            return 1  # turn_right (no-op, shouldn't happen)
        elif diff == 1:
            # Target is 1 turn right away
            return 1  # turn_right
        elif diff == 2:
            # Target is opposite (2 turns away, choose right arbitrarily)
            return 1  # turn_right
        else:  # diff == 3
            # Target is 1 turn left away (or 3 turns right)
            return 0  # turn_left

    def reset_worker_state(self):
        """Reset LSTM hidden state and traversal state."""
        self.hidden_state = None
        self.current_traversal_path = []
        self.traversal_step = 0
        self.current_edge_step = 0
        self.orientation_complete = False  # NEW: Reset orientation flag

    def compute_reward(self, current_state: Tuple[int, int], wide_goal: Tuple[int, int], narrow_goal: Tuple[int, int]) -> float:
        """
        Compute Worker's reward.
        Paper: "Worker receives rewards from Manager by reaching subgoals"
        """
        if current_state == narrow_goal:
            return 1.0  # Full success
        elif current_state == wide_goal:
            return 0.5  # Partial success (reached wide goal)
        else:
            return -0.001  # Step penalty
    
    # Transfer learning initialization (broken? just copying heads i guess)
    # def initialize_from_goal_policy(self, goal_policy):
    #     """
    #     Transfer learning: Initialize Worker with πg weights.
    #     Paper: "initializing task-specific Worker and Manager with weights from πg"
    #     """
    #     with torch.no_grad():
    #         # Initialize LSTM with small weights
    #         for name, param in self.lstm.named_parameters():
    #             if 'weight' in name:
    #                 nn.init.xavier_uniform_(param, gain=0.1)
    #             elif 'bias' in name:
    #                 nn.init.zeros_(param)
            
    #         # Copy actor head from goal policy if compatible
    #         if hasattr(goal_policy, 'actor') and goal_policy.actor.out_features == 7:
    #             # Goal policy has 7 actions, Worker uses 3 navigation actions
    #             # Copy first 3 action weights
    #             self.actor.weight.copy_(goal_policy.actor.weight[:3, :])
    #             self.actor.bias.copy_(goal_policy.actor.bias[:3])
    #         else:
    #             nn.init.xavier_uniform_(self.actor.weight, gain=0.1)
    #             nn.init.constant_(self.actor.bias, 0)
            
    #         # Copy critic
    #         if hasattr(goal_policy, 'critic'):
    #             self.critic.weight.copy_(goal_policy.critic.weight)
    #             self.critic.bias.copy_(goal_policy.critic.bias)
    #         else:
    #             nn.init.xavier_uniform_(self.critic.weight, gain=1.0)
    #             nn.init.constant_(self.critic.bias, 0)
        
    #     print(f"Worker initialized from goal policy (transfer learning)")

    #     # Transfer learning initialization from goal policy (copying LSTM and heads)
    
    # Actual working version:
    def initialize_from_goal_policy(self, goal_policy):
        with torch.no_grad():
            # Copy LSTM weights
            if hasattr(goal_policy, 'lstm'):
                # Goal policy LSTM: input=4, Worker LSTM: input=6
                # Copy what we can
                for name, param in goal_policy.lstm.named_parameters():
                    if name in dict(self.lstm.named_parameters()):
                        worker_param = dict(self.lstm.named_parameters())[name]
                        if param.shape == worker_param.shape:
                            worker_param.copy_(param)
                        elif 'weight_ih' in name:  # Input weights - partial copy
                            # Copy first 4 input dims (state_x, state_y, goal_x, goal_y)
                            worker_param[:, :4].copy_(param)
                            # Randomly init the extra 2 dims for narrow goal
                            nn.init.xavier_uniform_(worker_param[:, 4:])
                        else:  # Other weights match exactly
                            worker_param.copy_(param)
            
            # Copy actor (first 3 actions)
            if hasattr(goal_policy, 'actor'):
                self.actor.weight.copy_(goal_policy.actor.weight[:3, :])
                self.actor.bias.copy_(goal_policy.actor.bias[:3])
            
            # Copy critic
            if hasattr(goal_policy, 'critic'):
                self.critic.weight.copy_(goal_policy.critic.weight)
                self.critic.bias.copy_(goal_policy.critic.bias)
        
        print("Worker initialized from goal policy LSTM")


    def update_policy(self, states: List, actions: List, rewards: List, 
                     values: List[torch.Tensor], log_probs: List[torch.Tensor]):
        """
        Update Worker policy with per-step A2C.
        Paper: Worker operates at single-step resolution.
        """
        if len(rewards) == 0:
            return
        
        # Compute returns
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        
        # Handle single-step case
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if returns.dim() == 0:
            returns = returns.unsqueeze(0)
        
        # Compute advantages
        advantages = returns - values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Losses
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        
        # Entropy
        entropy_loss = 0
        for i, (state, wide_goal, narrow_goal) in enumerate(states):
            action_logits, _ = self.forward(state, wide_goal, narrow_goal)
            action_probs = F.softmax(action_logits, dim=0)
            entropy_loss += -(action_probs * torch.log(action_probs + 1e-8)).sum()
        entropy_loss = entropy_loss / len(states)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def generate_actions_from_path(self, path, current_agent_dir):
        """Convert position path to action sequence based on current orientation."""
        actions = []
        agent_dir = current_agent_dir
        
        for i in range(len(path) - 1):
            curr_pos = path[i]
            next_pos = path[i + 1]
            
            # Compute required direction from geometry
            dx = next_pos[0] - curr_pos[0]
            dy = next_pos[1] - curr_pos[1]
            
            if dx > 0:
                required_dir = 0  # Right
            elif dy > 0:
                required_dir = 1  # Down
            elif dx < 0:
                required_dir = 2  # Left
            elif dy < 0:
                required_dir = 3  # Up
            else:
                continue  # Same position, skip
            
            # Generate turns to face required direction
            while agent_dir != required_dir:
                diff = (required_dir - agent_dir) % 4
                if diff <= 2:
                    actions.append(1)  # turn_right
                    agent_dir = (agent_dir + 1) % 4
                else:
                    actions.append(0)  # turn_left
                    agent_dir = (agent_dir - 1) % 4
            
            # Move forward
            actions.append(2)
        
        return actions

class HierarchicalTrainer:
    def __init__(self, manager: HierarchicalManager, worker: HierarchicalWorker, 
                 env, horizon: int = 15,
                 diagnostic_interval: int = 30,
                 diagnostic_checkstart: bool = True,
                 workershaping=True,
                 managershaping=True):
        self.manager = manager
        self.worker = worker
        self.env = env
        self.horizon = horizon
        self.diagnostic_interval = diagnostic_interval
        self.diagnostic_checkstart = diagnostic_checkstart
        self.global_step_counter = 0
        self.global_episode_counter = 0
        
        # NEW: Diagnostic tracking
        self.diagnostic_history = {
            'manager_goal_diversity': [],
            'manager_entropy': [],
            'worker_goal_achievement': [],
            'balls_collected_per_episode': [],
            'manager_rewards_mean': [],
            'manager_rewards_std': [],
            'goal_distance_to_balls': [],
            'manager_value_mean': [],
            'worker_value_mean': [],
            'episode_rewards': []
        }

        self.worker_shaping_weight=0.2
        self.manager_shaping_weight=0.5
        self.manhattan_distance_rew_shaping=workershaping
        self.manager_reward_shaping=managershaping
    
    def train_episode(self, max_steps: int = 200, full_breakdown_every=1) -> Dict:
        """Train one episode with comprehensive diagnostics."""
        
        # Episode tracking
        episode_reward = 0
        episode_steps = 0
        manager_updates = 0
        worker_updates = 0

        # ADD THIS:
        all_manager_rewards_this_episode = []  # Track all horizon rewards for diagnostics

        # Reset environment and networks
        obs = self.env.reset()
        state = tuple(self.env.agent_pos)

        # ADD THIS DIAGNOSTIC HERE:
        if diag2:
            print(f"\n[EPISODE {self.global_episode_counter + 1} START]")
            print(f"  Agent at: {state}")
            print(f"  Balls at: {list(self.env.active_balls)}")
            print(f"  Pivotal states (first 5): {self.manager.pivotal_states[:5]}")

        optimal_reward, optimal_steps = compute_optimal_reward_for_episode(self.env)
        
        self.manager.reset_manager_state()
        self.worker.reset_worker_state()
        
        # Episode tracking
        episode_reward = 0
        episode_steps = 0
        manager_updates = 0
        worker_updates = 0
        manager_selection_counts = Counter()


        # NEW: Diagnostic tracking for this episode
        manager_entropies = []
        unique_manager_goals = set()
        manager_wide_goals_list = []
        manager_narrow_goals_list = []
        manager_values_list = []
        worker_values_list = []
        worker_goal_reached_count = 0
        total_horizons = 0
        
        # Manager experience accumulation
        manager_states = []
        manager_wide_goals = []
        manager_narrow_goals = []
        manager_rewards = []
        manager_values = []
        manager_log_probs = []
        manager_entropies_for_update = []
        
        horizon_counter = 0
        
        while episode_steps < max_steps:
            # Manager selects goals

            wide_goal, narrow_goal, manager_log_prob, manager_value = self.manager.get_manager_action(
                state, step_count=self.global_step_counter
            )
            
            manager_selection_counts[wide_goal] += 1

            # ADD THIS DIAGNOSTIC HERE:
            if diag2:
                balls_before_horizon = len(self.env.active_balls)
                if len(self.env.active_balls) > 0:
                    nearest_ball = min(self.env.active_balls, 
                                    key=lambda b: abs(wide_goal[0]-b[0]) + abs(wide_goal[1]-b[1]))
                    dist_to_nearest = abs(wide_goal[0]-nearest_ball[0]) + abs(wide_goal[1]-nearest_ball[1])
                    print(f"[MANAGER SELECT] wide={wide_goal}, narrow={narrow_goal}, "
                        f"nearest_ball={nearest_ball}, dist={dist_to_nearest}")

            # NEW: Track Manager diagnostics
            with torch.no_grad():
                wide_logits, _, _ = self.manager.forward(state)
                wide_probs = F.softmax(wide_logits, dim=0)
                entropy = -(wide_probs * torch.log(wide_probs + 1e-8)).sum()
                manager_entropies.append(entropy.item())
            
            unique_manager_goals.add(wide_goal)
            manager_wide_goals_list.append(wide_goal)
            manager_narrow_goals_list.append(narrow_goal)
            manager_values_list.append(manager_value.item())
            
            # ACCUMULATE Manager experience
            manager_states.append(state)
            manager_wide_goals.append(wide_goal)
            manager_narrow_goals.append(narrow_goal)
            manager_log_probs.append(manager_log_prob)  # ← REMOVE .detach()
            manager_values.append(manager_value)        # ← REMOVE .detach()
            manager_entropies_for_update.append(entropy.detach())  # Keep this
            
            # Worker executes for horizon steps
            worker_states = []
            worker_actions = []
            worker_rewards = []
            worker_values = []
            worker_log_probs = []
            
            horizon_env_reward = 0 
            goal_reached_this_horizon = False

            # Save starting state for Manager reward shaping
            starting_state_snapshot = state
            starting_balls_snapshot = list(self.env.active_balls)
            
            for h in range(self.horizon):
                # BEFORE taking action, record distance FOR SHAPING
                old_dist_narrow = manhattan_distance(state, narrow_goal)
                old_dist_wide = manhattan_distance(state, wide_goal)

                # Worker selects action
                action, worker_log_prob, worker_value = self.worker.get_action(
                    state, wide_goal, narrow_goal,
                    agent_dir=self.env.agent_dir  # ← ADD THIS
                )
                
                # NEW: Track Worker values
                worker_values_list.append(worker_value.item())
                
                # Environment step
                try:
                    obs, env_reward, terminated, truncated, info = self.env.step(action)
                    next_state = tuple(self.env.agent_pos)
                except (AssertionError, IndexError):
                    env_reward = -0.1
                    next_state = state
                    terminated = False
                    truncated = False
                
                if diag2:
                    if env_reward != 0:
                        print(f"[REWARD] Step {episode_steps}: env_reward={env_reward:.3f}, "
                            f"horizon_total={horizon_env_reward + env_reward:.3f}, "
                            f"agent_pos={next_state}, balls_remaining={len(self.env.active_balls)}")

                # Compute progress reward FOR SHAPING
                new_dist_narrow = manhattan_distance(next_state, narrow_goal)
                new_dist_wide = manhattan_distance(next_state, wide_goal)

                progress_narrow = (old_dist_narrow - new_dist_narrow) * 0.05
                progress_wide = (old_dist_wide - new_dist_wide) * 0.05
                progress_bonus = max(progress_narrow, progress_wide)  # Reward best progress

                # Worker reward (internal)
                worker_reward = self.worker.compute_reward(next_state, wide_goal, narrow_goal)
                # Add Manhattan Shaping

                if self.manhattan_distance_rew_shaping:
                    worker_reward += progress_bonus * self.worker_shaping_weight
                
                # NEW: Track if Worker reached goal
                if worker_reward > 0:
                    goal_reached_this_horizon = True
                
                # Store Worker experience
                worker_states.append((state, wide_goal, narrow_goal))
                worker_actions.append(action)
                worker_rewards.append(worker_reward)
                worker_values.append(worker_value)
                worker_log_probs.append(worker_log_prob)
                
                # # Update Worker every step
                # if len(worker_rewards) > 0:
                #     self.worker.update_policy(
                #         [worker_states[-1]], [worker_actions[-1]], [worker_rewards[-1]],
                #         [worker_values[-1]], [worker_log_probs[-1]]
                #     )
                #     worker_updates += 1
                
                # Track episode stats
                horizon_env_reward += env_reward
                episode_reward += env_reward
                
                
                self.global_step_counter += 1
                episode_steps += 1
                state = next_state
                
                if terminated or truncated:
                    break
            
            
            #  ADD: Update Worker AFTER horizon ends with full batch
            if len(worker_rewards) > 0:
                self.worker.update_policy(
                    worker_states,   # Full horizon: 20-30 samples
                    worker_actions,
                    worker_rewards,
                    worker_values,
                    worker_log_probs
                )
                worker_updates += 1


            if diag2:
                # ADD THIS DIAGNOSTIC HERE (after the for h in range loop):
                balls_collected_this_horizon = balls_before_horizon - len(self.env.active_balls)
                print(f"[HORIZON END] horizon_reward={horizon_env_reward:.3f}, "
                    f"balls_this_horizon={balls_collected_this_horizon}, "
                    f"wide_goal={wide_goal}, narrow_goal={narrow_goal}")
            
            # NEW: Track Worker success
            if goal_reached_this_horizon:
                worker_goal_reached_count += 1
            total_horizons += 1
            

            # Shaping

            if self.manager_reward_shaping:
                # AFTER - Add progress-based shaping:
                # Calculate how much closer we got to nearest ball during this horizon
                if len(self.env.active_balls) > 0:
                    balls_before_start = starting_balls_snapshot  # Save at horizon start
                    balls_after_end = list(self.env.active_balls)
                    
                    # Distance to nearest ball at horizon start vs end
                    start_pos_horizon = starting_state_snapshot
                    end_pos_horizon = state
                    
                    dist_before = min(manhattan_distance(start_pos_horizon, ball) 
                                    for ball in balls_before_start) if balls_before_start else 0
                    dist_after = min(manhattan_distance(end_pos_horizon, ball) 
                                    for ball in balls_after_end) if balls_after_end else 0
                    
                    progress_reward = (dist_before - dist_after) * self.manager_shaping_weight # Reward for getting closer
                    manager_reward = horizon_env_reward + progress_reward
                else:
                    manager_reward = horizon_env_reward
            else:
                manager_reward = horizon_env_reward

            # Manager collects reward
            manager_rewards.append(manager_reward)
            horizon_counter += 1


            MANAGER_UPDATE_FREQUENCY = 10

            # ADD THIS - save for diagnostics before resetting
            all_manager_rewards_this_episode.append(manager_reward)
                


            # Add after this line:
            if diag3:
                if horizon_counter % 10 == 0:  # Print every 10 horizons
                    print(f"  [REWARD DEBUG] Horizon {horizon_counter}: env_reward={horizon_env_reward:.3f}, total_so_far={sum(all_manager_rewards_this_episode):.3f}")

            
            # Update Manager after EVERY horizon
            if horizon_counter % MANAGER_UPDATE_FREQUENCY == 0 and len(manager_rewards) > 0:
                self.manager.update_policy(
                    manager_states, manager_wide_goals, manager_narrow_goals,
                    manager_rewards, manager_values, manager_log_probs,
                    manager_entropies_for_update,
                    step_count=self.global_step_counter
                )
                manager_updates += 1
                
                # ADD THIS: Detach AFTER update
                if self.manager.hidden_state is not None:
                    self.manager.hidden_state = tuple(h.detach() for h in self.manager.hidden_state)

                # Reset Manager experience
                manager_states = []
                manager_wide_goals = []
                manager_narrow_goals = []
                manager_rewards = []
                manager_values = []
                manager_log_probs = []
                manager_entropies_for_update = []
            
            if terminated or truncated:
                # ✅ ADD: Final update if any remaining experiences
                if len(manager_rewards) > 0:
                    self.manager.update_policy(
                        manager_states, manager_wide_goals, manager_narrow_goals,
                        manager_rewards, manager_values, manager_log_probs,
                        manager_entropies_for_update,
                        step_count=self.global_step_counter
                    )
                    manager_updates += 1
                    
                    if self.manager.hidden_state is not None:
                        self.manager.hidden_state = tuple(h.detach() for h in self.manager.hidden_state)
                break
        
        self.global_episode_counter += 1
        
        # # After episode
        # print(f"Manager selections: {manager_selection_counts.most_common(5)}")

        # NEW: Compute episode-level diagnostics
        balls_collected = self.env.total_balls - len(self.env.active_balls)
        worker_success_rate = worker_goal_reached_count / total_horizons if total_horizons > 0 else 0
        
        # Distance from Manager goals to balls
        avg_distance_to_balls = None
        if hasattr(self.env, 'active_balls') and len(self.env.active_balls) > 0:
            distances = []
            for wide_goal in manager_wide_goals_list:
                min_dist = min(
                    abs(wide_goal[0] - ball[0]) + abs(wide_goal[1] - ball[1])
                    for ball in self.env.active_balls
                )
                distances.append(min_dist)
            avg_distance_to_balls = np.mean(distances) if distances else None
        
        diversity_ratio = len(unique_manager_goals) / total_horizons if total_horizons > 0 else 0
        # Store diagnostics
        self.diagnostic_history['manager_goal_diversity'].append(diversity_ratio)
        self.diagnostic_history['manager_entropy'].append(np.mean(manager_entropies))
        self.diagnostic_history['worker_goal_achievement'].append(worker_success_rate)
        self.diagnostic_history['balls_collected_per_episode'].append(balls_collected)
        # Store diagnostics
        self.diagnostic_history['manager_rewards_mean'].append(
            np.mean(all_manager_rewards_this_episode) if all_manager_rewards_this_episode else 0.0
        )
        self.diagnostic_history['manager_rewards_std'].append(
            np.std(all_manager_rewards_this_episode) if len(all_manager_rewards_this_episode) > 1 else 0.0
        )
        
        if avg_distance_to_balls is not None:
            self.diagnostic_history['goal_distance_to_balls'].append(avg_distance_to_balls)
        self.diagnostic_history['manager_value_mean'].append(np.mean(manager_values_list))
        self.diagnostic_history['worker_value_mean'].append(np.mean(worker_values_list))
        self.diagnostic_history['episode_rewards'].append(episode_reward)
        
        # NEW: Print diagnostics every N episodes
        if self.global_episode_counter % full_breakdown_every == 0:
            print(f"\n{'#'*70}")
            print(f"Episode {self.global_episode_counter} Complete")
            print(f"{'#'*70}")
            print(f"Task Performance:")
            print(f"  Episode reward: {episode_reward:.2f} (optimal: {optimal_reward:.2f})")
            print(f"  Balls collected: {balls_collected}/{self.env.total_balls}")
            print(f"  Episode steps: {episode_steps}")
            print(f"\nManager Diagnostics:")
            print(f"  Goal diversity: {len(unique_manager_goals)}/{len(self.manager.pivotal_states)} unique goals")
            print(f"  Avg entropy: {np.mean(manager_entropies):.3f}")
            
            if len(all_manager_rewards_this_episode) > 0:
                print(f"  Avg reward: {np.mean(all_manager_rewards_this_episode):.3f} ± {np.std(all_manager_rewards_this_episode):.3f}")
            else:
                print(f"  Avg reward: N/A (no data)")

            print(f"  Avg value estimate: {np.mean(manager_values_list):.3f}")
            if avg_distance_to_balls is not None:
                print(f"  Avg distance to balls: {avg_distance_to_balls:.1f}")
            print(f"\nWorker Diagnostics:")
            print(f"  Goal achievement rate: {worker_success_rate*100:.1f}%")
            print(f"  Avg value estimate: {np.mean(worker_values_list):.3f}")
            print(f"\nTraining Stats:")
            print(f"  Manager updates: {manager_updates}")
            print(f"  Worker updates: {worker_updates}")
            print(f"{'#'*70}\n")
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'manager_updates': manager_updates,
            'worker_updates': worker_updates,
            'manager_entropy': np.mean(manager_entropies) if manager_entropies else 0,
            'final_entropy': manager_entropies[-1] if manager_entropies else 0,
            'unique_manager_goals': len(unique_manager_goals),
            'goal_diversity_history': None,
            'optimal_reward': optimal_reward,
            'optimal_steps': optimal_steps
        }
