import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import random
import numpy as np


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
        self.entropy_coef = 0.01
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
        
    def initialize_from_goal_policy(self, goal_policy):
        """
        Transfer learning: Initialize Manager with better scaling.
        """
        with torch.no_grad():
            # Use standard initialization (gain=1.0, not 0.1)
            for name, param in self.lstm.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param, gain=1.0)  # ← FIX: gain=1.0
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param, gain=1.0)  # ← Better for recurrent
                elif 'bias' in name:
                    nn.init.zeros_(param)
            
            # Initialize output heads with standard gain
            nn.init.xavier_uniform_(self.wide_head.weight, gain=1.0)
            nn.init.zeros_(self.wide_head.bias)
            nn.init.xavier_uniform_(self.narrow_head.weight, gain=1.0)
            nn.init.zeros_(self.narrow_head.bias)
            
            # Initialize critic
            if hasattr(goal_policy, 'critic'):
                self.critic.weight.copy_(goal_policy.critic.weight)
                self.critic.bias.copy_(goal_policy.critic.bias)
            else:
                nn.init.xavier_uniform_(self.critic.weight, gain=1.0)
                nn.init.zeros_(self.critic.bias)
        
        print(f"Manager initialized with standard scaling (gain=1.0)")

    def update_policy(self, states, wide_goals, narrow_goals, rewards, values, log_probs, entropies, step_count=0):
        """
        Update Manager policy with comprehensive diagnostics.
        
        Args:
            step_count: Global step counter for conditional printing
        """
        if len(rewards) == 0:
            return
        

        # Configurable diagnostic printing
        if self.diagnostic_checkstart and step_count < 15:
            verbose = True
        elif step_count % self.diagnostic_interval == 0:
            verbose = True
        else:
            verbose = False
            verbose = False
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"[Manager Update Debug] Step {step_count}")
            print(f"{'='*70}")
        
        # Convert to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values_tensor = torch.stack(values).squeeze()
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)

        # Fix dimension mismatch
        if values_tensor.dim() == 0:
            values_tensor = values_tensor.unsqueeze(0)
        if rewards_tensor.dim() == 0:
            rewards_tensor = rewards_tensor.unsqueeze(0)
        
        if verbose:
            print(f"Batch size: {len(rewards)}")
            print(f"Rewards: {rewards_tensor.tolist()}")
            print(f"Values: {values_tensor.tolist()}")
            print(f"Log probs: {log_probs_tensor.tolist()}")
            print(f"Entropies: {entropies_tensor.tolist()}")

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        if verbose:
            print(f"Returns (discounted): {returns.tolist()}")
        
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
        
        if verbose:
            print(f"Advantages (raw): min={advantages.min().item():.3f}, max={advantages.max().item():.3f}, mean={advantages.mean().item():.3f}")
        
        # Normalize advantages
        advantages_prenorm = advantages.clone()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            if verbose:
                print(f"Advantages (normalized): min={advantages.min().item():.3f}, max={advantages.max().item():.3f}, mean={advantages.mean().item():.3f}")
        else:
            if verbose:
                print(f"Advantages: Single sample, no normalization applied")

        # Store pre-update weights for comparison
        if verbose:
            pre_weights = {}
            for name, param in self.named_parameters():
                if 'lstm' in name or 'wide_head' in name:
                    pre_weights[name] = param.data.clone()

        # Policy loss
        policy_loss = -(advantages.detach() * log_probs_tensor).mean()

        # Value loss
        value_loss = F.mse_loss(values_tensor, returns)

        # Entropy (pre-computed)
        avg_entropy = entropies_tensor.mean()

        if verbose:
            print(f"\nLoss components:")
            print(f"  Policy loss: {policy_loss.item():.6f}")
            print(f"  Value loss: {value_loss.item():.6f}")
            print(f"  Avg entropy: {avg_entropy.item():.6f}")
            print(f"  Entropy coef: {self.entropy_coef}")
            print(f"  Value coef: {self.value_coef}")

        # Combined loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * avg_entropy

        if verbose:
            print(f"  Total loss: {total_loss.item():.6f}")

        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients before clipping
        if verbose:
            print(f"\nGradients (before clipping):")
            total_grad_norm = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    if grad_norm > 0.01 or 'lstm' in name or 'wide_head' in name:
                        print(f"  {name}: {grad_norm:.6f}")
            total_grad_norm = total_grad_norm ** 0.5
            print(f"  Total grad norm: {total_grad_norm:.6f}")
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        # Check gradients after clipping
        if verbose:
            print(f"\nGradients (after clipping):")
            total_grad_norm_clipped = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm_clipped += grad_norm ** 2
                    if grad_norm > 0.01 or 'lstm' in name or 'wide_head' in name:
                        print(f"  {name}: {grad_norm:.6f}")
            total_grad_norm_clipped = total_grad_norm_clipped ** 0.5
            print(f"  Total grad norm (clipped): {total_grad_norm_clipped:.6f}")
        
        self.optimizer.step()
        
        # Check weight changes
        if verbose:
            print(f"\nWeight changes:")
            for name, param in self.named_parameters():
                if name in pre_weights:
                    weight_change = (param.data - pre_weights[name]).norm().item()
                    weight_norm = param.data.norm().item()
                    print(f"  {name}: change={weight_change:.6f}, norm={weight_norm:.6f}")
        
        if verbose:
            print(f"{'='*70}\n")

# Test functions for HierarchicalManager
def test_manager_architecture():
    """Test Manager network architecture and forward pass."""
    print("Testing Manager Architecture:")
    print("=" * 40)
    
    # Sample pivotal states
    pivotal_states = [(2, 3), (5, 7), (8, 1), (4, 6), (7, 9)]
    
    # Create manager
    manager = HierarchicalManager(
        pivotal_states=pivotal_states,
        neighborhood_size=3,
        horizon=15
    )
    
    print(f"Manager created with {len(pivotal_states)} pivotal states")
    print(f"Total parameters: {sum(p.numel() for p in manager.parameters())}")
    
    # Test forward pass
    test_state = (4, 5)
    print(f"\nTesting forward pass with state: {test_state}")
    
    wide_logits, features, value = manager.forward(test_state)
    print(f"Wide logits shape: {wide_logits.shape}")
    print(f"Features shape: {features.shape}")  
    print(f"Value shape: {value.shape}")
    print(f"Value estimate: {value.item():.3f}")
    
    print("\n" + "=" * 40)
    print("Manager Architecture test completed!")

def test_wide_goal_selection():
    """Test wide goal selection mechanism."""
    print("Testing Wide Goal Selection:")
    print("=" * 40)
    
    pivotal_states = [(1, 2), (3, 4), (5, 6), (7, 8)]
    manager = HierarchicalManager(pivotal_states=pivotal_states)
    
    test_state = (2, 3)
    print(f"Agent state: {test_state}")
    print(f"Available pivotal states: {pivotal_states}")
    
    # Test multiple selections
    selections = {}
    for trial in range(10):
        wide_idx, wide_log_prob, value = manager.select_wide_goal(test_state)
        wide_goal = pivotal_states[wide_idx]
        
        if wide_goal not in selections:
            selections[wide_goal] = 0
        selections[wide_goal] += 1
        
        if trial < 3:  # Print first few
            print(f"Trial {trial+1}: Selected {wide_goal} (idx={wide_idx}, log_prob={wide_log_prob:.3f})")
    
    print(f"\nSelection distribution over 10 trials:")
    for goal, count in selections.items():
        print(f"  {goal}: {count} times")
    
    print("\n" + "=" * 40)
    print("Wide Goal Selection test completed!")

def test_narrow_goal_selection():
    """Test narrow goal selection in neighborhood."""
    print("Testing Narrow Goal Selection:")
    print("=" * 40)
    
    pivotal_states = [(5, 5)]  # Single pivotal state for testing
    manager = HierarchicalManager(pivotal_states=pivotal_states, neighborhood_size=3)
    
    agent_state = (3, 4)
    wide_goal = (5, 5)
    
    print(f"Agent state: {agent_state}")
    print(f"Wide goal: {wide_goal}")
    
    # Get neighborhood
    neighborhood = manager.get_neighborhood(wide_goal)
    print(f"3×3 Neighborhood: {neighborhood}")
    
    # Test narrow selections
    selections = {}
    for trial in range(15):
        narrow_goal, narrow_log_prob = manager.select_narrow_goal(agent_state, wide_goal)
        
        if narrow_goal not in selections:
            selections[narrow_goal] = 0
        selections[narrow_goal] += 1
        
        if trial < 3:
            print(f"Trial {trial+1}: Selected {narrow_goal} (log_prob={narrow_log_prob:.3f})")
    
    print(f"\nNarrow goal distribution over 15 trials:")
    for goal, count in selections.items():
        print(f"  {goal}: {count} times")
    
    print("\n" + "=" * 40)
    print("Narrow Goal Selection test completed!")

def test_complete_manager_action():
    """Test complete Wide-then-Narrow action selection."""
    print("Testing Complete Manager Action:")
    print("=" * 40)
    
    pivotal_states = [(2, 2), (6, 3), (4, 7), (8, 5)]
    manager = HierarchicalManager(pivotal_states=pivotal_states)
    
    agent_state = (5, 4)
    print(f"Agent state: {agent_state}")
    print(f"Pivotal states: {pivotal_states}")
    
    # Test complete action selection
    for trial in range(3):
        print(f"\nTrial {trial+1}:")
        wide_goal, narrow_goal, combined_log_prob, value = manager.get_manager_action(agent_state)
        
        print(f"  Wide goal: {wide_goal}")
        print(f"  Narrow goal: {narrow_goal}")
        print(f"  Combined log_prob: {combined_log_prob:.3f}")
        print(f"  Value estimate: {value.item():.3f}")
        
        # Verify narrow goal is in neighborhood of wide goal
        neighborhood = manager.get_neighborhood(wide_goal)
        is_valid = narrow_goal in neighborhood
        print(f"  Narrow goal in wide neighborhood: {is_valid}")
    
    print("\n" + "=" * 40)
    print("Complete Manager Action test completed!")

def test_manager_with_goal_policy():
    """Test Manager initialization from goal policy."""
    print("Testing Manager Transfer Learning:")
    print("=" * 40)
    
    # Mock goal policy for testing
    class MockGoalPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )
            self.actor = nn.Linear(64, 7)
            self.critic = nn.Linear(64, 1)
    
    mock_policy = MockGoalPolicy()
    print(f"Mock policy parameters: {sum(p.numel() for p in mock_policy.parameters())}")
    
    # Create manager and initialize
    pivotal_states = [(1, 1), (5, 5), (9, 9)]
    manager = HierarchicalManager(pivotal_states=pivotal_states)
    
    print(f"Manager parameters before init: {sum(p.numel() for p in manager.parameters())}")
    
    # Test initialization
    manager.initialize_from_goal_policy(mock_policy)
    
    # Test that manager still works after initialization
    test_state = (3, 3)
    wide_goal, narrow_goal, log_prob, value = manager.get_manager_action(test_state)
    
    print(f"After initialization - Action: wide={wide_goal}, narrow={narrow_goal}")
    print(f"Log prob: {log_prob:.3f}, Value: {value.item():.3f}")
    
    print("\n" + "=" * 40)
    print("Manager Transfer Learning test completed!")

def hierarchical_manager_tests():
    """Run all Hierarchical Manager tests."""
    print("\n" + "#" * 60)
    test_manager_architecture()
    test_wide_goal_selection()
    test_narrow_goal_selection()
    test_complete_manager_action()
    test_manager_with_goal_policy()

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
        self.verbose=verbose
        super().__init__()
        
        self.device = device
        self.world_graph = world_graph
        self.pivotal_states = set(pivotal_states)  # For fast membership check
        
        # A2C-LSTM architecture (paper: "Manager and Worker are both A2C-LSTMs")
        self.lstm = nn.LSTM(
            input_size=6,  # [state_x, state_y, gw_x, gw_y, gn_x, gn_y]
            hidden_size=64,
            num_layers=1,
            batch_first=True
        ).to(device)
        
        # Action head: 3 navigation actions [turn_left, turn_right, move_forward]
        self.actor = nn.Linear(64, 3).to(device)
        
        # Value function
        self.critic = nn.Linear(64, 1).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Worker state
        self.hidden_state = None
        self.current_traversal_path = []  # Active graph traversal
        self.traversal_step = 0
        
        # Hyperparameters
        self.gamma = 0.99
        self.entropy_coef = 0.01
        self.value_coef = 0.5
    
    def reset_worker_state(self):
        """Reset LSTM hidden state and traversal."""
        self.hidden_state = None
        self.current_traversal_path = []
        self.traversal_step = 0
    
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
    
    def get_action(self, state: Tuple[int, int], wide_goal: Tuple[int, int], narrow_goal: Tuple[int, int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from Worker policy with graph traversal integration.
        Paper: "Worker can traverse the world via graph if at pivotal state"
        """
        # Check if should use graph traversal
        if self.should_traverse(state, wide_goal) and not self.current_traversal_path:
            # Initiate new traversal
            self.current_traversal_path = self.plan_traversal(state, wide_goal)
            self.traversal_step = 0
            
            if self.current_traversal_path:
                if self.verbose:
                    print(f"    Worker initiating graph traversal: {state} -> {wide_goal}")
        
        # If actively traversing, follow graph path
        if self.current_traversal_path and self.traversal_step < len(self.current_traversal_path) - 1:
            # Get next position in path
            current_pos = self.current_traversal_path[self.traversal_step]
            next_pos = self.current_traversal_path[self.traversal_step + 1]
            
            # Get stored action sequence from graph
            edge_actions = self.world_graph.get_edge_actions(current_pos, next_pos)
            
            if edge_actions and self.traversal_step < len(edge_actions):
                # Use actual stored action
                action = edge_actions[self.traversal_step]
            else:
                # Fallback to move_forward if no actions stored
                action = 2
            
            self.traversal_step += 1
            
            # Check if traversal complete
            if self.traversal_step >= len(self.current_traversal_path) - 1:
                if self.verbose:
                    print(f"    Worker completed graph traversal to {wide_goal}")
                self.current_traversal_path = []
                self.traversal_step = 0
            
            # Get value estimate from network
            with torch.no_grad():
                _, value = self.forward(state, wide_goal, narrow_goal)
            
            # Dummy log prob for traversal actions
            log_prob = torch.tensor(-1.0, device=self.device)
            
            return action, log_prob, value.squeeze()
        
        # Normal policy action
        with torch.no_grad():
            action_logits, value = self.forward(state, wide_goal, narrow_goal)
        
        # Sample from action distribution
        action_probs = F.softmax(action_logits, dim=0)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value.squeeze()
    
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
    
    def initialize_from_goal_policy(self, goal_policy):
        """
        Transfer learning: Initialize Worker with πg weights.
        Paper: "initializing task-specific Worker and Manager with weights from πg"
        """
        with torch.no_grad():
            # Initialize LSTM with small weights
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            
            # Copy actor head from goal policy if compatible
            if hasattr(goal_policy, 'actor') and goal_policy.actor.out_features == 7:
                # Goal policy has 7 actions, Worker uses 3 navigation actions
                # Copy first 3 action weights
                self.actor.weight.copy_(goal_policy.actor.weight[:3, :])
                self.actor.bias.copy_(goal_policy.actor.bias[:3])
            else:
                nn.init.xavier_uniform_(self.actor.weight, gain=0.1)
                nn.init.constant_(self.actor.bias, 0)
            
            # Copy critic
            if hasattr(goal_policy, 'critic'):
                self.critic.weight.copy_(goal_policy.critic.weight)
                self.critic.bias.copy_(goal_policy.critic.bias)
            else:
                nn.init.xavier_uniform_(self.critic.weight, gain=1.0)
                nn.init.constant_(self.critic.bias, 0)
        
        print(f"Worker initialized from goal policy (transfer learning)")

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

# Test functions for HierarchicalWorker
def test_worker_architecture():
    """Test Worker network architecture."""
    print("Testing Worker Architecture:")
    print("=" * 40)
    
    # Mock world graph
    from utils.graph_manager import GraphManager
    world_graph = GraphManager()
    pivotal_states = [(2, 2), (5, 5), (8, 8)]
    for state in pivotal_states:
        world_graph.add_node(state)
    world_graph.add_edge((2, 2), (5, 5), 5)
    world_graph.add_edge((5, 5), (8, 8), 5)
    
    # Create worker
    worker = HierarchicalWorker(world_graph, pivotal_states)
    
    print(f"Worker created with {len(pivotal_states)} pivotal states")
    print(f"Total parameters: {sum(p.numel() for p in worker.parameters())}")
    
    # Test forward pass
    state = (3, 3)
    wide_goal = (5, 5)
    narrow_goal = (5, 6)
    
    action_logits, value = worker.forward(state, wide_goal, narrow_goal)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value.item():.3f}")
    
    print("\n" + "=" * 40)
    print("Worker Architecture test completed!")

def test_worker_action_selection():
    """Test Worker action selection."""
    print("Testing Worker Action Selection:")
    print("=" * 40)
    
    from utils.graph_manager import GraphManager
    world_graph = GraphManager()
    pivotal_states = [(1, 1), (5, 5)]
    for state in pivotal_states:
        world_graph.add_node(state)
    
    worker = HierarchicalWorker(world_graph, pivotal_states)
    
    state = (3, 3)
    wide_goal = (5, 5)
    narrow_goal = (6, 5)
    
    print(f"State: {state}, Wide goal: {wide_goal}, Narrow goal: {narrow_goal}")
    
    # Test action selection
    for trial in range(3):
        action, log_prob, value = worker.get_action(state, wide_goal, narrow_goal)
        action_names = ['turn_left', 'turn_right', 'move_forward']
        print(f"Trial {trial+1}: {action_names[action]}, log_prob={log_prob:.3f}, value={value.item():.3f}")
    
    print("\n" + "=" * 40)
    print("Worker Action Selection test completed!")

def test_graph_traversal():
    """Test Worker graph traversal functionality."""
    print("Testing Worker Graph Traversal:")
    print("=" * 40)
    
    from utils.graph_manager import GraphManager
    world_graph = GraphManager()
    pivotal_states = [(2, 2), (5, 5), (8, 8)]
    
    # Build connected graph
    for state in pivotal_states:
        world_graph.add_node(state)
    world_graph.add_edge((2, 2), (5, 5), 5)
    world_graph.add_edge((5, 5), (8, 8), 5)
    
    worker = HierarchicalWorker(world_graph, pivotal_states)
    
    # Test traversal planning
    print(f"Pivotal states: {pivotal_states}")
    
    current = (2, 2)
    target = (8, 8)
    
    print(f"\nPlanning traversal: {current} -> {target}")
    path = worker.plan_traversal(current, target)
    print(f"Planned path: {path}")
    
    # Test should_traverse
    should_traverse = worker.should_traverse(current, target)
    print(f"Should traverse: {should_traverse}")
    
    # Test traversal execution
    print(f"\nSimulating traversal from {current}:")
    worker.current_traversal_path = path
    worker.traversal_step = 0
    
    for step in range(len(path)):
        action, log_prob, value = worker.get_action(current, target, target)
        print(f"  Step {step+1}: action={action}, traversing={len(worker.current_traversal_path) > 0}")
        if worker.traversal_step < len(path):
            current = path[worker.traversal_step]
    
    print("\n" + "=" * 40)
    print("Worker Graph Traversal test completed!")

def test_worker_rewards():
    """Test Worker reward computation."""
    print("Testing Worker Rewards:")
    print("=" * 40)
    
    from utils.graph_manager import GraphManager
    world_graph = GraphManager()
    worker = HierarchicalWorker(world_graph, [(5, 5)])
    
    wide_goal = (5, 5)
    narrow_goal = (6, 5)
    
    # Test different reward scenarios
    scenarios = [
        ((6, 5), "Reached narrow goal"),
        ((5, 5), "Reached wide goal"),
        ((3, 3), "Neither goal")
    ]
    
    for state, description in scenarios:
        reward = worker.compute_reward(state, wide_goal, narrow_goal)
        print(f"{description}: state={state}, reward={reward}")
    
    print("\n" + "=" * 40)
    print("Worker Rewards test completed!")

def test_worker_transfer_learning():
    """Test Worker initialization from goal policy."""
    print("Testing Worker Transfer Learning:")
    print("=" * 40)
    
    # Mock goal policy
    class MockGoalPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Linear(64, 7)
            self.critic = nn.Linear(64, 1)
    
    mock_policy = MockGoalPolicy()
    
    from utils.graph_manager import GraphManager
    world_graph = GraphManager()
    worker = HierarchicalWorker(world_graph, [(1, 1)])
    
    print("Initializing Worker from goal policy...")
    worker.initialize_from_goal_policy(mock_policy)
    
    # Test Worker still works
    action, log_prob, value = worker.get_action((2, 2), (5, 5), (6, 5))
    print(f"After initialization: action={action}, log_prob={log_prob:.3f}, value={value.item():.3f}")
    
    print("\n" + "=" * 40)
    print("Worker Transfer Learning test completed!")

def hierarchical_worker_tests():   
    """Run all Hierarchical Worker tests."""
    print("\n" + "#" * 60)
    test_worker_architecture()
    test_worker_action_selection()
    test_graph_traversal()
    test_worker_rewards()
    test_worker_transfer_learning()

class HierarchicalTrainer:
    """
    Coordinates Manager and Worker training.
    Paper: "Hierarchical reinforcement learning framework that leverages world graph"
    """
    
    def __init__(self, manager: HierarchicalManager, worker: HierarchicalWorker, 
                 env, horizon: int = 15,
                 diagnostic_interval: int = 30,
                 diagnostic_checkstart: bool = True):
        self.manager = manager
        self.worker = worker
        self.env = env
        self.horizon = horizon
        self.diagnostic_interval = diagnostic_interval
        self.diagnostic_checkstart = diagnostic_checkstart
        self.global_step_counter = 0  # NEW: Global counter across ALL episodes
        self.global_episode_counter = 0
    
    def train_episode(self, max_steps: int = 200, full_breakdown_every=1) -> Dict:
        """Train one episode with comprehensive diagnostics."""
        # Reset environment and networks
        obs = self.env.reset()
        state = tuple(self.env.agent_pos)
        self.manager.reset_manager_state()
        self.worker.reset_worker_state()
        
        # Episode tracking
        episode_reward = 0
        episode_steps = 0
        manager_updates = 0
        worker_updates = 0
        
        # Diagnostic tracking
        manager_entropies = []
        unique_manager_goals = set()
        goal_diversity_history = []
        
        # Manager experience accumulation
        manager_states = []
        manager_wide_goals = []
        manager_narrow_goals = []
        manager_rewards = []
        manager_values = []
        manager_log_probs = []
        manager_entropies_for_update = []
        
        horizon_counter = 0
        
        # print(f"\n{'#'*70}")
        # print(f"Starting Episode - max_steps={max_steps}, horizon={self.horizon}")
        # print(f"{'#'*70}")
        
        while episode_steps < max_steps:
            # Configurable diagnostic printing using GLOBAL counter
            if self.diagnostic_checkstart and self.global_step_counter < 15:
                verbose_action = True
            elif self.global_step_counter % self.diagnostic_interval == 0:
                verbose_action = True
            else:
                verbose_action = False
            
            if verbose_action:
                print(f"\n--- Global Step {self.global_step_counter} (Episode Step {episode_steps}, Horizon {horizon_counter}) ---")
            
            # Manager selects goals - PASS GLOBAL COUNTER
            wide_goal, narrow_goal, manager_log_prob, manager_value = self.manager.get_manager_action(
                state, step_count=self.global_step_counter
            )
            
            # CRITICAL: Compute entropy with CURRENT hidden state
            with torch.no_grad():
                saved_hidden = self.manager.hidden_state
                wide_logits_saved, _, _ = self.manager.forward(state)
                wide_probs_saved = F.softmax(wide_logits_saved, dim=0)
                entropy_saved = -(wide_probs_saved * torch.log(wide_probs_saved + 1e-8)).sum()
                
                # DIAGNOSTIC: Compare with reset hidden state (use parameter, not hardcoded)
                if verbose_action and self.global_step_counter % self.diagnostic_interval == 0:
                    self.manager.hidden_state = None
                    wide_logits_reset, _, _ = self.manager.forward(state)
                    wide_probs_reset = F.softmax(wide_logits_reset, dim=0)
                    
                    logit_diff = (wide_logits_saved - wide_logits_reset).abs().max()
                    prob_diff = (wide_probs_saved - wide_probs_reset).abs().max()
                    
                    print(f"\n[Hidden State Diagnostic]")
                    print(f"  Logit diff (saved vs reset): {logit_diff.item():.6f}")
                    print(f"  Prob diff (saved vs reset): {prob_diff.item():.6f}")
                    print(f"  Top 3 logits (saved): {wide_logits_saved[:3].tolist()}")
                    print(f"  Top 3 logits (reset): {wide_logits_reset[:3].tolist()}")
                    
                    if logit_diff > 1.0:
                        print(f"  WARNING: Large hidden state desynchronization!")
                
                self.manager.hidden_state = saved_hidden
                manager_entropy = entropy_saved
            
            # Track diagnostics
            manager_entropies.append(manager_entropy.item())
            unique_manager_goals.add(wide_goal)
            goal_diversity_history.append(len(unique_manager_goals))
            
            # Detach Manager's hidden state
            if self.manager.hidden_state is not None:
                self.manager.hidden_state = tuple(h.detach() for h in self.manager.hidden_state)
            
            # ACCUMULATE Manager experience
            manager_states.append(state)
            manager_wide_goals.append(wide_goal)
            manager_narrow_goals.append(narrow_goal)
            manager_log_probs.append(manager_log_prob)
            manager_values.append(manager_value)
            manager_entropies_for_update.append(manager_entropy)
            
            # Worker executes for horizon steps
            worker_states = []
            worker_actions = []
            worker_rewards = []
            worker_values = []
            worker_log_probs = []
            
            horizon_env_reward = 0
            goal_reached = False
            
            for h in range(self.horizon):
                # Worker selects action
                action, worker_log_prob, worker_value = self.worker.get_action(state, wide_goal, narrow_goal)
                
                # Environment step
                try:
                    obs, env_reward, terminated, truncated, info = self.env.step(action)
                    next_state = tuple(self.env.agent_pos)
                except (AssertionError, IndexError):
                    env_reward = -0.1
                    next_state = state
                    terminated = False
                    truncated = False
                
                # Worker reward (internal)
                worker_reward = self.worker.compute_reward(next_state, wide_goal, narrow_goal)
                
                # Store Worker experience
                worker_states.append((state, wide_goal, narrow_goal))
                worker_actions.append(action)
                worker_rewards.append(worker_reward)
                worker_values.append(worker_value)
                worker_log_probs.append(worker_log_prob)
                
                # Update Worker every step
                if len(worker_rewards) > 0:
                    self.worker.update_policy(
                        [worker_states[-1]], [worker_actions[-1]], [worker_rewards[-1]],
                        [worker_values[-1]], [worker_log_probs[-1]]
                    )
                    worker_updates += 1
                
                # Track episode stats
                horizon_env_reward += env_reward
                episode_reward += env_reward
                
                # INCREMENT GLOBAL COUNTER
                self.global_step_counter += 1
                episode_steps += 1
                state = next_state
                
                # Check goal reached
                if worker_reward > 0:
                    goal_reached = True
                    if verbose_action:
                        print(f"  Worker reached goal at step {self.global_step_counter}")
                    break
                
                if terminated or truncated:
                    break
            
            # Manager receives horizon reward
            manager_rewards.append(horizon_env_reward)
            horizon_counter += 1
            
            if verbose_action:
                print(f"  Horizon {horizon_counter} complete: reward={horizon_env_reward:.3f}, steps={self.global_step_counter}")
            
            # Update Manager after EVERY horizon
            if len(manager_rewards) > 0:
                self.manager.update_policy(
                    manager_states, manager_wide_goals, manager_narrow_goals,
                    manager_rewards, manager_values, manager_log_probs,
                    manager_entropies_for_update,
                    step_count=self.global_step_counter
                )
                manager_updates += 1
                
                # Reset Manager experience
                manager_states = []
                manager_wide_goals = []
                manager_narrow_goals = []
                manager_rewards = []
                manager_values = []
                manager_log_probs = []
                manager_entropies_for_update = []
            
            if terminated or truncated:
                break
        
        self.global_episode_counter += 1

        if self.global_episode_counter%full_breakdown_every == 0: 
            # Episode summary

            print(f"\n{'#'*70}")
            print(f"Episode",self.global_episode_counter,"Complete")
            print(f"{'#'*70}")
            print(f"Total reward: {episode_reward:.2f}")
            print(f"Total steps: {episode_steps}")
            print(f"Manager updates: {manager_updates}")
            print(f"Worker updates: {worker_updates}")
            print(f"Unique goals selected: {len(unique_manager_goals)}/{len(self.manager.pivotal_states)}")
            print(f"Final entropy: {manager_entropies[-1]:.3f}")
            print(f"Average entropy: {np.mean(manager_entropies):.3f}")
            print(f"{'#'*70}\n")
        
        return {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'manager_updates': manager_updates,
            'worker_updates': worker_updates,
            'manager_entropy': np.mean(manager_entropies) if manager_entropies else 0,
            'final_entropy': manager_entropies[-1] if manager_entropies else 0,
            'unique_manager_goals': len(unique_manager_goals),
            'goal_diversity_history': goal_diversity_history,
        }


#This test uses a MockEnv to simulate environment interactions, not the Minigrid Wrapper
def test_hierarchical_training_loop():
    """Test complete hierarchical training loop."""
    print("Testing Hierarchical Training Loop:")
    print("=" * 40)
    
    # Mock environment
    class MockEnv:
        def __init__(self):
            self.agent_pos = (2, 2)
            self.step_count = 0
        
        def reset(self):
            self.agent_pos = (2, 2)
            self.step_count = 0
            return None
        
        def step(self, action):
            self.step_count += 1
            # Simple movement simulation
            if action == 2:  # move_forward
                self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
            
            reward = 0.1 if self.step_count > 10 else 0
            done = self.step_count >= 20
            return None, reward, done, False, {}
    
    # Create components
    from utils.graph_manager import GraphManager
    world_graph = GraphManager()
    pivotal_states = [(2, 2), (5, 5), (8, 8)]
    for state in pivotal_states:
        world_graph.add_node(state)
    
    manager = HierarchicalManager(pivotal_states, horizon=5)
    worker = HierarchicalWorker(world_graph, pivotal_states)
    env = MockEnv()
    
    trainer = HierarchicalTrainer(manager, worker, env, horizon=5)
    
    print("Training single episode...")
    stats = trainer.train_episode(max_steps=20)
    
    print(f"Episode reward: {stats['episode_reward']:.2f}")
    print(f"Episode steps: {stats['episode_steps']}")
    print(f"Manager updates: {stats['manager_updates']}")
    print(f"Worker updates: {stats['worker_updates']}")
    
    print("\n" + "=" * 40)
    print("Hierarchical Training Loop test completed!")

def hierarchical_system_tests():
    """Run all hierarchical system tests."""
    print("\n" + "#" * 60)
    print("HIERARCHICAL SYSTEM TESTS")
    print("#" * 60)
    hierarchical_manager_tests()
    hierarchical_worker_tests()
    test_hierarchical_training_loop()

# Integration test with real environment
def test_phase2_with_real_environment():
    """Integration test: Phase 2 HRL with actual MinigridWrapper and Phase 1 outputs."""
    print("Testing Phase 2 with Real Environment:")
    print("=" * 60)
    
    from wrappers.minigrid_wrapper import MinigridWrapper, EnvSizes, EnvModes
    from local_networks.vaesystem import VAESystem
    from local_networks.policy_networks import GoalConditionedPolicy
    from utils.statistics_buffer import StatBuffer
    
    # Setup Phase 1 environment
    env = MinigridWrapper(size=EnvSizes.SMALL, mode=EnvModes.MULTIGOAL)
    env.phase = 1
    env.randomgen = False
    
    print("Phase 1: Discovering world graph...")
    policy = GoalConditionedPolicy(lr=5e-3)
    vae_system = VAESystem(state_dim=16, action_vocab_size=7, mu0=3.0, grid_size=env.size)
    buffer = StatBuffer()
    
    # Quick alternating training (reduced for testing)
    pivotal_states, world_graph = run_quick_phase1(env, policy, vae_system, buffer, max_iterations=3)
    
    print(f"\nPhase 1 complete: {len(pivotal_states)} pivotal states, {len(world_graph.edges)} edges")
    
    # Initialize Phase 2 components
    print("\nPhase 2: Initializing hierarchical system...")
    manager = HierarchicalManager(pivotal_states, neighborhood_size=3, horizon=10)
    worker = HierarchicalWorker(world_graph, pivotal_states)
    
    # Transfer learning
    manager.initialize_from_goal_policy(policy)
    worker.initialize_from_goal_policy(policy)
    
    # Switch to Phase 2
    env.phase = 2
    trainer = HierarchicalTrainer(manager, worker, env, horizon=10)
    
    # Train episodes
    print("\nTraining hierarchical policy on MULTIGOAL task...")
    for episode in range(3):
        stats = trainer.train_episode(max_steps=100)
        print(f"Episode {episode+1}: reward={stats['episode_reward']:.2f}, steps={stats['episode_steps']}")
    
    print("\n" + "=" * 60)

# Simplified Phase 1 function for testing
def run_quick_phase1(env, policy, vae_system, buffer, max_iterations=3):
    """Simplified Phase 1 for testing."""
    pivotal_states = []
    
    for iteration in range(max_iterations):
        obs = env.reset()
        start_pos = tuple(env.agent_pos)
        
        try:
            episodes = policy.collect_episodes_from_position(
                env, start_pos, num_episodes=2, vae_system=vae_system
            )
            if episodes:
                buffer.add_episodes(episodes)
        except:
            pass
        
        if buffer.episodes_in_buffer >= 3:
            try:
                pivotal_states = vae_system.train(buffer, num_epochs=10, batch_size=2)
            except:
                pass
    
    # Build world graph
    if len(pivotal_states) > 0:
        world_graph = policy.complete_world_graph_discovery(env, pivotal_states)
    else:
        from utils.graph_manager import GraphManager
        world_graph = GraphManager()
        pivotal_states = [(2, 2), (5, 5), (7, 7)]
        for state in pivotal_states:
            world_graph.add_node(state)
    
    return pivotal_states, world_graph