import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Optional
import numpy as np

# PROJECT-SPECIFIC IMPORTS
from collections import Counter
from utils.misc import manhattan_distance

diag=False
diag2=False
diag3=False

class HierarchicalManager(nn.Module):
    """
    HIERARCHICAL MANAGER: SELECTS GOALS USING WIDE-THEN-NARROW STRATEGY
    """
    
    def __init__(self, 
                 pivotal_states: List[Tuple[int, int]],
                 neighborhood_size: int = 3,
                 lr: float = 5e-3,
                 horizon: int = 15,
                 diagnostic_interval: int = 30,  # NEW
                 diagnostic_checkstart: bool = True,  # NEW
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # INITIALIZE MANAGER NETWORKS AND PARAMETERS
        super().__init__()
        
        self.device = device
        self.pivotal_states = pivotal_states
        self.neighborhood_size = neighborhood_size
        self.horizon = horizon
        
    # A2C-LSTM ARCHITECTURE
        self.lstm = nn.LSTM(
            input_size=4,  # [state_x, state_y, prev_gw_x, prev_gw_y] 
            hidden_size=64,
            num_layers=1,
            batch_first=True
        ).to(device)
        
    # WIDE POLICY OUTPUT LAYER
        self.wide_head = nn.Linear(64, len(pivotal_states)).to(device)
        
    # NARROW POLICY OUTPUT LAYER
        self.narrow_head = nn.Linear(64 + 2, (2 * neighborhood_size) ** 2).to(device)  # +2 for gw coords; neighborhood_size = radius
        
    # VALUE FUNCTION OUTPUT LAYER
        self.critic = nn.Linear(64, 1).to(device)
        
    # OPTIMIZER
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    # MANAGER STATE VARIABLES
        self.hidden_state = None
        self.prev_wide_goal = (0, 0)  # Previous wide goal for LSTM input
        
    # HYPERPARAMETERS
        self.gamma = 0.99
        self.entropy_coef = 1e-5
        self.value_coef = 0.05

    # DIAGNOSTIC PARAMETERS
        self.diagnostic_interval = diagnostic_interval
        self.diagnostic_checkstart = diagnostic_checkstart
        self._valid_cells = None
    
    def reset_manager_state(self):
        # RESET MANAGER STATE
        self.hidden_state = None
        self.prev_wide_goal = (0, 0)
    
    def get_neighborhood(self, wide_goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        # neighborhood_size = radius → (2r)×(2r) = 144 cells for r=6
        gw_x, gw_y = wide_goal
        neighborhood = []
        r = self.neighborhood_size
        for dx in range(-r, r):
            for dy in range(-r, r):
                neighborhood.append((gw_x + dx, gw_y + dy))
        return neighborhood
    
    def forward(self, state: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FORWARD PASS: WIDE GOAL SELECTION
    # PREPARE LSTM INPUT
        lstm_input = torch.tensor([
            state[0], state[1], 
            self.prev_wide_goal[0], self.prev_wide_goal[1]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        
    # LSTM FORWARD PASS
        lstm_out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)
        features = lstm_out.squeeze(0).squeeze(0)  # [64]
        
    # WIDE POLICY OUTPUT
        wide_logits = self.wide_head(features)  # [num_pivotal_states]
        
    # VALUE FUNCTION OUTPUT
        value = self.critic(features)  # [1]
        
        return wide_logits, features, value
    
    def select_wide_goal(self, state: Tuple[int, int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        # SELECT WIDE GOAL FROM PIVOTAL STATES
        wide_logits, features, value = self.forward(state)
        
    # SAMPLE FROM CATEGORICAL DISTRIBUTION
        wide_probs = F.softmax(wide_logits, dim=0)
        wide_dist = torch.distributions.Categorical(wide_probs)
        wide_idx = wide_dist.sample()
        wide_log_prob = wide_dist.log_prob(wide_idx)
        
    # GET PIVOTAL STATE COORDINATES
        wide_goal = self.pivotal_states[wide_idx.item()]
        
    # UPDATE PREVIOUS WIDE GOAL
        self.prev_wide_goal = wide_goal
        
        return wide_idx.item(), wide_log_prob, value.squeeze()
    
    def _nearest_valid_cell(self, center: Tuple[int, int], valid_cells: set) -> Tuple[int, int]:
        return min(valid_cells, key=lambda c: abs(c[0] - center[0]) + abs(c[1] - center[1]))

    def select_narrow_goal(self, state: Tuple[int, int], wide_goal: Tuple[int, int], valid_cells=None) -> Tuple[Tuple[int, int], torch.Tensor]:
        _, features, _ = self.forward(state)
        wide_goal_tensor = torch.tensor(wide_goal, dtype=torch.float32, device=self.device)
        narrow_input = torch.cat([features, wide_goal_tensor])
        narrow_logits = self.narrow_head(narrow_input)

        neighborhood = self.get_neighborhood(wide_goal)

        if valid_cells is not None:
            valid_indices = [i for i, cell in enumerate(neighborhood) if cell in valid_cells]
            if not valid_indices:
                return self._nearest_valid_cell(wide_goal, valid_cells), torch.tensor(0.0, device=self.device)

            idx_t = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
            valid_logits = narrow_logits[idx_t]
            valid_probs = F.softmax(valid_logits, dim=0)
            valid_dist = torch.distributions.Categorical(valid_probs)
            local_idx = valid_dist.sample()
            log_prob = valid_dist.log_prob(local_idx)
            narrow_goal = neighborhood[valid_indices[local_idx.item()]]
            return narrow_goal, log_prob

        narrow_probs = F.softmax(narrow_logits, dim=0)
        dist = torch.distributions.Categorical(narrow_probs)
        idx = dist.sample()
        return neighborhood[idx.item()], dist.log_prob(idx)
    
    def get_manager_action(self, state: Tuple[int, int], step_count: int = 0, valid_cells=None):
        verbose = (self.diagnostic_checkstart and step_count < 15) or (step_count % self.diagnostic_interval == 0)

        if verbose:
            print(f"\n[Manager Action] Step {step_count}")
            print(f"  Current state: {state}")
            if self.hidden_state is not None:
                h, c = self.hidden_state
                print(f"  Hidden state norms: h={h.norm().item():.3f}, c={c.norm().item():.3f}")

        # Pass 1: wide goal — also captures logits/entropy without an extra forward
        wide_logits, _, value = self.forward(state)
        wide_probs = F.softmax(wide_logits, dim=0)
        entropy = -(wide_probs * torch.log(wide_probs + 1e-8)).sum()

        wide_dist = torch.distributions.Categorical(wide_probs)
        wide_idx = wide_dist.sample()
        wide_log_prob = wide_dist.log_prob(wide_idx)
        wide_goal = self.pivotal_states[wide_idx.item()]
        self.prev_wide_goal = wide_goal  # updated before pass 2

        if verbose:
            top_k = min(5, len(wide_logits))
            top_logits, top_indices = wide_logits.topk(top_k)
            max_entropy = torch.log(torch.tensor(float(len(self.pivotal_states))))
            print(f"  Wide goal selection:")
            print(f"    Top {top_k} logits: {top_logits.tolist()}")
            print(f"    Top {top_k} probs: {wide_probs[top_indices].tolist()}")
            print(f"    Selected idx: {wide_idx.item()}, goal: {wide_goal}")
            print(f"    Entropy: {entropy.item():.3f}/{max_entropy.item():.3f} ({100*entropy/max_entropy:.1f}%)")

        # Pass 2: narrow goal — LSTM now sees updated prev_wide_goal as context
        self._valid_cells = valid_cells
        narrow_goal, narrow_log_prob = self.select_narrow_goal(state, wide_goal, valid_cells=valid_cells)
        combined_log_prob = wide_log_prob + narrow_log_prob

        if verbose:
            print(f"  Narrow goal: {narrow_goal}")
            print(f"  Combined log prob: {combined_log_prob.item():.3f}")
            print(f"  Value: {value.squeeze().item():.3f}")

        return wide_goal, narrow_goal, combined_log_prob, value.squeeze(), entropy
        
    # TRANSFER LEARNING INITIALIZATION (COMMENTED OUT)
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

    # INITIALIZE FROM GOAL POLICY (LSTM COPYING)
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
        
        # Skip single-sample batches (can't normalize advantages)
        if len(rewards) <= 1:
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
        raw_advantages = advantages.clone()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.clamp(-3.0, 3.0)

        # Policy and value losses
        policy_loss = -(advantages.detach() * log_probs_tensor).mean()
        value_loss = F.mse_loss(values_tensor, returns)

        if verbose:
            print(f"  Advantages (raw): {raw_advantages[:3].tolist()}")
            print(f"  Advantages (normalized): {advantages[:3].tolist()}")
            print(f"  Policy loss: {policy_loss.item():.4f}")
        
        # Compute narrow entropy — save/restore hidden state so the loop
        # does not corrupt the recurrent state used outside this update
        saved_hidden = tuple(h.detach().clone() for h in self.hidden_state) if self.hidden_state else None
        narrow_entropies = []
        for state_i, wide_goal_i in zip(states, wide_goals):
            _, features_i, _ = self.forward(state_i)
            wide_goal_tensor = torch.tensor(wide_goal_i, dtype=torch.float32, device=self.device)
            narrow_input = torch.cat([features_i, wide_goal_tensor])
            narrow_logits = self.narrow_head(narrow_input)
            if self._valid_cells is not None:
                neighborhood_i = self.get_neighborhood(wide_goal_i)
                valid_idx = [j for j, c in enumerate(neighborhood_i) if c in self._valid_cells]
                if valid_idx:
                    idx_t = torch.tensor(valid_idx, dtype=torch.long, device=self.device)
                    narrow_logits = narrow_logits[idx_t]
            narrow_probs = F.softmax(narrow_logits, dim=0)
            narrow_entropy = -(narrow_probs * torch.log(narrow_probs + 1e-8)).sum()
            narrow_entropies.append(narrow_entropy)
        self.hidden_state = saved_hidden
        
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
                 verbose: bool = False,
                 goal_policy=None,
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

        # Phase 1 navigator — fine-tuned during Phase 2.
        # Stored via object.__setattr__ to prevent PyTorch from registering it as a
        # submodule: otherwise self.parameters() would include GCP params, causing
        # double-counting in the optimizer and polluting worker.state_dict().
        object.__setattr__(self, 'goal_policy', goal_policy)

        # A2C-LSTM architecture
        self.lstm = nn.LSTM(
            input_size=6,  # [state_x, state_y, gw_x, gw_y, gn_x, gn_y]
            hidden_size=64,
            num_layers=1,
            batch_first=True
        ).to(device)
        
        self.actor = nn.Linear(64, 3).to(device)
        self.critic = nn.Linear(64, 1).to(device)
        
        if self.goal_policy is not None:
            # Fine-tune GCP at 1/10 the worker LR to preserve Phase 1 knowledge
            self.optimizer = optim.Adam([
                {'params': self.parameters(), 'lr': lr},
                {'params': self.goal_policy.parameters(), 'lr': lr * 0.1},
            ])
        else:
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
        if self.goal_policy is not None:
            self.goal_policy.hidden_state = None
    
    def is_at_pivotal_state(self, state: Tuple[int, int]) -> bool:
        """Check if current state is a pivotal state."""
        return state in self.pivotal_states
    
    def plan_traversal(self, current_state: Tuple[int, int], target_state: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Plan graph traversal from current pivotal state toward target pivotal state.
        First tries a direct Dijkstra path; if unreachable, falls back to the closest
        reachable pivotal node to target (best-effort traversal).
        """
        if not self.is_at_pivotal_state(current_state):
            return None
        if not self.is_at_pivotal_state(target_state):
            return None

        # Primary: exact path to target
        path, _ = self.world_graph.shortest_path(current_state, target_state)
        if path and len(path) > 1:
            return path

        # Best-effort: route to closest reachable pivotal state to target
        reachable = self.world_graph.get_reachable_nodes(current_state)
        reachable.discard(current_state)
        if not reachable:
            return None

        best_node = min(reachable,
                        key=lambda n: abs(n[0] - target_state[0]) + abs(n[1] - target_state[1]))
        path, _ = self.world_graph.shortest_path(current_state, best_node)
        if path and len(path) > 1:
            return path

        return None

    def should_traverse(self, current_state: Tuple[int, int], wide_goal: Tuple[int, int]) -> bool:
        """
        Determine if Worker should initiate graph traversal.
        Paper: "Worker can traverse via world graph if it encounters pivotal state g'w with feasible connection to gw"
        """
        if not self.is_at_pivotal_state(current_state):
            return False
        if current_state == wide_goal:
            return False
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
                self._traversal_starts_this_episode = getattr(self, '_traversal_starts_this_episode', 0) + 1
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

        # 3. FALLBACK: fine-tune Phase 1 GCP if available, else use worker A2C.
        if self.goal_policy is not None:
            # Worker value: uses wide+narrow goal context
            _, value = self.forward(state, wide_goal, narrow_goal)

            # GCP selects action toward narrow_goal — gradients flow for fine-tuning
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
            gn_t = torch.tensor(narrow_goal, dtype=torch.float32, device=self.device)
            gp_logits, _ = self.goal_policy.forward(state_t, gn_t)
            if gp_logits.dim() > 1:
                gp_logits = gp_logits.squeeze(0)
            masked = torch.full_like(gp_logits, float('-inf'))
            masked[[0, 1, 2]] = gp_logits[[0, 1, 2]]  # navigation actions only
            probs = F.softmax(masked, dim=-1)
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
            action = idx.item()
        else:
            action_logits, value = self.forward(state, wide_goal, narrow_goal)
            probs = F.softmax(action_logits, dim=0)
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            log_prob = dist.log_prob(idx)
            action = idx.item()

        return action, log_prob, value.squeeze()

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
        
        # Entropy — computed on the network that actually selects actions
        entropy_loss = 0
        if self.goal_policy is not None:
            saved_hs = self.goal_policy.hidden_state
            self.goal_policy.hidden_state = None
            for (state, wide_goal, narrow_goal) in states:
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
                gn_t = torch.tensor(narrow_goal, dtype=torch.float32, device=self.device)
                gp_logits, _ = self.goal_policy.forward(state_t, gn_t)
                if gp_logits.dim() > 1:
                    gp_logits = gp_logits.squeeze(0)
                masked = torch.full_like(gp_logits, float('-inf'))
                masked[[0, 1, 2]] = gp_logits[[0, 1, 2]]
                action_probs = F.softmax(masked, dim=-1)
                entropy_loss += -(action_probs * torch.log(action_probs + 1e-8)).sum()
            self.goal_policy.hidden_state = saved_hs
        else:
            for (state, wide_goal, narrow_goal) in states:
                action_logits, _ = self.forward(state, wide_goal, narrow_goal)
                action_probs = F.softmax(action_logits, dim=0)
                entropy_loss += -(action_probs * torch.log(action_probs + 1e-8)).sum()
        entropy_loss = entropy_loss / len(states)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.goal_policy is not None:
            all_params = list(self.parameters()) + list(self.goal_policy.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)
        else:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Truncated BPTT: detach hidden states so graphs don't grow across horizons
        if self.hidden_state is not None:
            self.hidden_state = tuple(h.detach() for h in self.hidden_state)
        if self.goal_policy is not None and self.goal_policy.hidden_state is not None:
            self.goal_policy.hidden_state = tuple(h.detach() for h in self.goal_policy.hidden_state)

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
                 managershaping=True,
                 narrow_shaping_weight: float = 1.0,
                 goal_timeout: int = 3):
        self.manager = manager
        self.worker = worker
        self.env = env
        self.horizon = horizon
        self.diagnostic_interval = diagnostic_interval
        self.diagnostic_checkstart = diagnostic_checkstart
        self.goal_timeout = goal_timeout  # max horizons before forcing a new goal
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

        self.worker_shaping_weight=0.2   # max ~0.15/horizon << success reward 1.0
        self.manager_shaping_weight=1
        self.narrow_shaping_weight=narrow_shaping_weight
        self.manhattan_distance_rew_shaping=workershaping
        self.manager_reward_shaping=managershaping
    
    def train_episode(self, max_steps: int = 200, full_breakdown_every=1):
        """Train one episode with comprehensive diagnostics."""

        # Episode tracking
        episode_reward = 0
        episode_steps = 0
        manager_updates = 0
        worker_updates = 0

        all_manager_rewards_this_episode = []  # Track all horizon rewards for diagnostics

        # Reset environment and networks
        obs = self.env.reset()
        state = tuple(self.env.agent_pos)
        valid_cells = {
            (x, y)
            for x in range(self.env.width)
            for y in range(self.env.height)
            if self.env._is_traversable(self.env.grid.get(x, y))
        }

        if diag2:
            print(f"\n[EPISODE {self.global_episode_counter + 1} START]")
            print(f"  Agent at: {state}")
            print(f"  Balls at: {list(self.env.active_balls)}")
            print(f"  Pivotal states (first 5): {self.manager.pivotal_states[:5]}")

        self.manager.reset_manager_state()
        self.worker.reset_worker_state()
        self.worker._traversal_starts_this_episode = 0

        # Goal persistence state
        active_wide_goal = None
        active_narrow_goal = None
        active_log_prob = None
        active_value = None
        active_entropy = None
        horizons_on_goal = 0
        goal_reached_prev = False   # did the worker reach the goal last horizon?
        ball_collected_prev = False  # was a ball collected last horizon?

        # Episode tracking
        episode_reward = 0
        episode_steps = 0
        manager_updates = 0
        worker_updates = 0
        manager_selection_counts = Counter()
        traversal_starts = 0  # how many times graph traversal is initiated


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
            # ── GOAL SELECTION (persistence) ──────────────────────────────────
            need_new_goal = (
                active_wide_goal is None
                or goal_reached_prev
                or ball_collected_prev
                or horizons_on_goal >= self.goal_timeout
            )

            if need_new_goal:
                if active_wide_goal is not None:
                    # flush only traversal state — LSTM context stays valid across goals
                    self.worker.current_traversal_path = []
                    self.worker.traversal_step = 0
                    self.worker.current_edge_actions = None
                    self.worker.current_action_idx = 0

                wide_goal, narrow_goal, manager_log_prob, manager_value, entropy = self.manager.get_manager_action(
                    state, step_count=self.global_step_counter, valid_cells=valid_cells
                )
                if self.manager.hidden_state is not None:
                    self.manager.hidden_state = tuple(h.detach() for h in self.manager.hidden_state)

                active_wide_goal = wide_goal
                active_narrow_goal = narrow_goal
                active_log_prob = manager_log_prob
                active_value = manager_value
                active_entropy = entropy.detach()
                horizons_on_goal = 0
            else:
                wide_goal = active_wide_goal
                narrow_goal = active_narrow_goal

            manager_selection_counts[wide_goal] += 1
            manager_entropies.append(active_entropy.item())

            if diag2:
                balls_before_horizon = len(self.env.active_balls)
                if len(self.env.active_balls) > 0:
                    nearest_ball = min(self.env.active_balls,
                                    key=lambda b: abs(wide_goal[0]-b[0]) + abs(wide_goal[1]-b[1]))
                    dist_to_nearest = abs(wide_goal[0]-nearest_ball[0]) + abs(wide_goal[1]-nearest_ball[1])
                    print(f"[MANAGER SELECT] wide={wide_goal}, narrow={narrow_goal}, "
                        f"nearest_ball={nearest_ball}, dist={dist_to_nearest}")

            unique_manager_goals.add(wide_goal)
            manager_wide_goals_list.append(wide_goal)
            manager_narrow_goals_list.append(narrow_goal)
            manager_values_list.append(active_value.item())
            
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
                    agent_dir=self.env.agent_dir
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
                
                if next_state == narrow_goal:
                    goal_reached_this_horizon = True
                
                # Store Worker experience
                # Only store if worker is NOT traversing
                if not self.worker.current_traversal_path:
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
            balls_collected_this_horizon = len(starting_balls_snapshot) - len(self.env.active_balls)
            manager_reward = horizon_env_reward

            if self.manager_reward_shaping:
                # Bonus for ball collection
                manager_reward += balls_collected_this_horizon * 5
                
                # Narrow-goal closeness shaping: prefer narrow goals nearer to balls
                if len(starting_balls_snapshot) > 0:
                    narrow_dist_to_ball = min(
                        manhattan_distance(narrow_goal, ball)
                        for ball in starting_balls_snapshot
                    )
                    narrow_bonus = self.narrow_shaping_weight / (1.0 + narrow_dist_to_ball)
                    manager_reward += narrow_bonus

                # Distance-based progress shaping (fixed)
                if len(starting_balls_snapshot) > 0:
                    # Only compare distances to balls that STILL EXIST
                    remaining_balls = [b for b in starting_balls_snapshot if b in self.env.active_balls]
                    
                    if len(remaining_balls) > 0:
                        start_pos_horizon = starting_state_snapshot
                        end_pos_horizon = state
                        
                        # Distance to same set of balls before/after
                        dist_before = min(manhattan_distance(start_pos_horizon, ball) for ball in remaining_balls)
                        dist_after = min(manhattan_distance(end_pos_horizon, ball) for ball in remaining_balls)
                        
                        progress = dist_before - dist_after
                        progress_reward = progress * self.manager_shaping_weight
                        manager_reward += progress_reward

            # Bonus for choosing a reachable narrow goal close to a ball
            if goal_reached_this_horizon:
                narrow_bonus = 0.2
                if len(starting_balls_snapshot) > 0:
                    dist_narrow_to_ball = min(
                        manhattan_distance(narrow_goal, ball) for ball in starting_balls_snapshot
                    )
                    narrow_bonus += 2.0 / (1.0 + dist_narrow_to_ball)
                manager_reward += narrow_bonus

            # Push manager experience every horizon
            manager_states.append(starting_state_snapshot)
            manager_wide_goals.append(active_wide_goal)
            manager_narrow_goals.append(active_narrow_goal)
            manager_rewards.append(manager_reward)
            manager_values.append(active_value)
            manager_log_probs.append(active_log_prob)
            manager_entropies_for_update.append(active_entropy)

            horizons_on_goal += 1
            horizon_counter += 1
            goal_reached_prev = goal_reached_this_horizon
            ball_collected_prev = balls_collected_this_horizon > 0

            all_manager_rewards_this_episode.append(manager_reward)

            if diag3:
                if horizon_counter % 10 == 0:
                    print(f"  [REWARD DEBUG] Horizon {horizon_counter}: env_reward={horizon_env_reward:.3f}, total_so_far={sum(all_manager_rewards_this_episode):.3f}")

            if terminated or truncated:
                break

        # ── FINAL MANAGER UPDATE ──────────────────────────────────────────────
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
            print(f"  Episode reward: {episode_reward:.2f}")
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
            print(f"  Graph traversals initiated: {self.worker._traversal_starts_this_episode}")
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
            'balls_collected': balls_collected,
        }
