import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import random


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
    
    def get_manager_action(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int], torch.Tensor, torch.Tensor]:
        """
        Complete Wide-then-Narrow goal selection.
        Paper: "Manager first selects a pivotal state and then specifies a final goal within nearby neighborhood"
        
        Returns:
            wide_goal: Selected pivotal state
            narrow_goal: Selected local goal
            combined_log_prob: Combined log probability
            value: State value estimate
        """
        # Step 1: Wide goal selection
        wide_idx, wide_log_prob, value = self.select_wide_goal(state)
        wide_goal = self.pivotal_states[wide_idx]
        
        # Step 2: Narrow goal selection  
        narrow_goal, narrow_log_prob = self.select_narrow_goal(state, wide_goal)
        
        # Combined log probability: log(π^ω × π^n)
        combined_log_prob = wide_log_prob + narrow_log_prob
        
        return wide_goal, narrow_goal, combined_log_prob, value
    
    def initialize_from_goal_policy(self, goal_policy):
        """
        Transfer learning: Initialize Manager with πg weights.
        Paper: "implicit skill transfer by initializing task-specific Worker and Manager with weights from πg"
        
        Note: Manager uses LSTM while πg uses linear layers, so we initialize with small
        random weights to maintain similar learned structure conceptually.
        """
        with torch.no_grad():
            # Initialize LSTM with small random weights (can't directly copy from linear layers)
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=0.1)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            
            # Initialize output heads with small weights
            nn.init.xavier_uniform_(self.wide_head.weight, gain=0.1)
            nn.init.constant_(self.wide_head.bias, 0)
            nn.init.xavier_uniform_(self.narrow_head.weight, gain=0.1)
            nn.init.constant_(self.narrow_head.bias, 0)
            
            # Initialize critic similarly to goal policy's critic if available
            if hasattr(goal_policy, 'critic'):
                self.critic.weight.copy_(goal_policy.critic.weight)
                self.critic.bias.copy_(goal_policy.critic.bias)
            else:
                nn.init.xavier_uniform_(self.critic.weight, gain=1.0)
                nn.init.constant_(self.critic.bias, 0)
        
        print(f"Manager initialized from goal policy (transfer learning)")

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
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            world_graph: GraphManager with edges between pivotal states
            pivotal_states: List of pivotal state coordinates
            lr: Learning rate
            device: Computing device
        """
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
                print(f"    Worker initiating graph traversal: {state} -> {wide_goal}")
        
        # If actively traversing, follow graph path
        if self.current_traversal_path and self.traversal_step < len(self.current_traversal_path) - 1:
            # Get next position in path
            current_pos = self.current_traversal_path[self.traversal_step]
            next_pos = self.current_traversal_path[self.traversal_step + 1]
            
            # Determine action needed (simplified: assumes agent faces correct direction)
            # In reality, would use stored edge action sequences
            action = 2  # move_forward (placeholder)
            
            self.traversal_step += 1
            
            # Check if traversal complete
            if self.traversal_step >= len(self.current_traversal_path) - 1:
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
            return -0.01  # Step penalty
    
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