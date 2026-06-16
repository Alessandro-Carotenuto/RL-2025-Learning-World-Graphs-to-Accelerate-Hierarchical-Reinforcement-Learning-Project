#----------------------------------------------------------------------------#
#                            IMPORTS & MODULE SETUP                          #
#----------------------------------------------------------------------------#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np

# PROJECT-SPECIFIC IMPORTS
from collections import Counter
from utils.misc import manhattan_distance

diag=False
diag2=False
diag3=False

#----------------------------------------------------------------------------#
#                            HIERARCHICAL MANAGER                            #
#----------------------------------------------------------------------------#

class HierarchicalManager(nn.Module):
    """
    HIERARCHICAL MANAGER: SELECTS GOALS USING WIDE-THEN-NARROW STRATEGY
    """
    
    def __init__(self,
                 pivotal_states: List[Tuple[int, int]],
                 neighborhood_size: int = 3,
                 lr: float = 5e-3,
                 horizon: int = 15,
                 diagnostic_interval: int = 30,
                 diagnostic_checkstart: bool = True,
                 action_verbose: bool = False,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        # INITIALIZE MANAGER NETWORKS AND PARAMETERS
        super().__init__()
        
        self.device = device
        self.pivotal_states = pivotal_states
        self.neighborhood_size = neighborhood_size
        self.horizon = horizon
        
    # A2C-LSTM ARCHITECTURE
        self.lstm = nn.LSTM(
            input_size=7,  # [state_x, state_y, prev_gw_x, prev_gw_y, nearest_ball_x, nearest_ball_y, n_remaining]
            hidden_size=64,
            num_layers=1,
            batch_first=True
        ).to(device)
        
    # WIDE POLICY OUTPUT LAYER
        self.wide_head = nn.Linear(64, len(pivotal_states)).to(device)
        
    # NARROW POLICY OUTPUT LAYER — stateless MLP, input = [dx, dy] nearest ball rel. to wide_goal
        narrow_out = 2 * neighborhood_size * (neighborhood_size + 1) + 1
        self.narrow_head = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, narrow_out),
        ).to(device)
        
    # VALUE FUNCTION OUTPUT LAYER
        self.critic = nn.Linear(64, 1).to(device)
        
    # OPTIMIZER
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    # MANAGER STATE VARIABLES
        self.hidden_state = None
        self.prev_wide_goal = (0, 0)  # Previous wide goal for LSTM input
        
    # HYPERPARAMETERS
        self.gamma = 0.99
        self.entropy_coef = 0.001
        self.value_coef = 0.2

    # DIAGNOSTIC PARAMETERS
        self.diagnostic_interval = diagnostic_interval
        self.diagnostic_checkstart = diagnostic_checkstart
        self.action_verbose = action_verbose
        self._valid_cells = None
    
    def reset_manager_state(self):
        # RESET MANAGER STATE
        self.hidden_state = None
        self.prev_wide_goal = (0, 0)
    
    def get_neighborhood(self, wide_goal: Tuple[int, int], valid_cells=None) -> List[Tuple[int, int]]:
        # Manhattan diamond incl. center: all (dx,dy) with |dx|+|dy| <= r → 2r(r+1)+1 cells
        gw_x, gw_y = wide_goal
        r = self.neighborhood_size
        neighborhood = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    cell = (gw_x + dx, gw_y + dy)
                    if valid_cells is None or cell in valid_cells:
                        neighborhood.append(cell)
        return neighborhood
    
    def forward(self, state: Tuple[int, int], active_balls=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FORWARD PASS: WIDE GOAL SELECTION
    # PREPARE LSTM INPUT
        if active_balls and len(active_balls) > 0:
            nearest = min(active_balls, key=lambda b: abs(b[0] - state[0]) + abs(b[1] - state[1]))
            nb_x, nb_y, n_remaining = float(nearest[0]), float(nearest[1]), float(len(active_balls))
        else:
            nb_x, nb_y, n_remaining = 0.0, 0.0, 0.0
        lstm_input = torch.tensor([
            state[0], state[1],
            self.prev_wide_goal[0], self.prev_wide_goal[1],
            nb_x, nb_y, n_remaining
        ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        
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

    def select_narrow_goal(self, state: Tuple[int, int], wide_goal: Tuple[int, int], valid_cells=None, active_balls=None, narrow_blacklist=None) -> Tuple[Tuple[int, int], torch.Tensor]:
        if active_balls:
            nearest = min(active_balls, key=lambda b: abs(b[0] - wide_goal[0]) + abs(b[1] - wide_goal[1]))
            dx, dy = float(nearest[0] - wide_goal[0]), float(nearest[1] - wide_goal[1])
        else:
            dx, dy = 0.0, 0.0
        narrow_logits = self.narrow_head(torch.tensor([dx, dy], dtype=torch.float32, device=self.device))

        # Always build the full diamond first so logit i always maps to full_neighborhood[i]
        neighborhood_full = self.get_neighborhood(wide_goal)

        if valid_cells is not None:
            valid_indices = [i for i, cell in enumerate(neighborhood_full) if cell in valid_cells]
            if not valid_indices:
                return self._nearest_valid_cell(wide_goal, valid_cells), torch.tensor(0.0, device=self.device)

            # Exclude blacklisted cells; fallback to all valid if all are blacklisted
            if narrow_blacklist:
                candidates = [i for i in valid_indices if neighborhood_full[i] not in narrow_blacklist]
                if candidates:
                    valid_indices = candidates

            idx_t = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
            valid_logits = narrow_logits[idx_t]
            valid_probs = F.softmax(valid_logits, dim=0)
            valid_dist = torch.distributions.Categorical(valid_probs)
            local_idx = valid_dist.sample()
            log_prob = valid_dist.log_prob(local_idx)
            narrow_goal = neighborhood_full[valid_indices[local_idx.item()]]
            return narrow_goal, log_prob

        # No valid_cells: mask blacklisted indices directly in logits
        if narrow_blacklist:
            non_blacklisted = [i for i, cell in enumerate(neighborhood_full) if cell not in narrow_blacklist]
            if non_blacklisted:  # fallback: if all cells blacklisted, ignore blacklist
                for i, cell in enumerate(neighborhood_full):
                    if cell in narrow_blacklist:
                        narrow_logits[i] = -1e9

        narrow_probs = F.softmax(narrow_logits, dim=0)
        dist = torch.distributions.Categorical(narrow_probs)
        idx = dist.sample()
        return neighborhood_full[idx.item()], dist.log_prob(idx)
    
    def get_manager_action(self, state: Tuple[int, int], step_count: int = 0, valid_cells=None, active_balls=None, wide_blacklist=None, narrow_blacklist=None, temperature: float = 1.0):
        verbose = self.action_verbose and (
            (self.diagnostic_checkstart and step_count < 15) or
            (step_count % self.diagnostic_interval == 0)
        )

        if verbose:
            print(f"\n[Manager Action] Step {step_count}")
            print(f"  Current state: {state}")
            if self.hidden_state is not None:
                h, c = self.hidden_state
                print(f"  Hidden state norms: h={h.norm().item():.3f}, c={c.norm().item():.3f}")

        # Pass 1: wide goal — also captures logits/entropy without an extra forward
        wide_logits, _, value = self.forward(state, active_balls)

        # Mask blacklisted wide goals; fallback: ignore blacklist if all pivots would be masked
        if wide_blacklist:
            blacklisted = [i for i, ps in enumerate(self.pivotal_states) if ps in wide_blacklist]
            if len(blacklisted) < len(self.pivotal_states):
                for i in blacklisted:
                    wide_logits[i] = -1e9

        wide_probs = F.softmax(wide_logits / temperature, dim=0)
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
        narrow_goal, narrow_log_prob = self.select_narrow_goal(state, wide_goal, valid_cells=valid_cells, active_balls=active_balls, narrow_blacklist=narrow_blacklist)
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
            nn.init.xavier_uniform_(self.narrow_head[-1].weight, gain=0.5)
            nn.init.zeros_(self.narrow_head[-1].bias)
        
        print("Manager initialized from goal policy LSTM")


    def update_policy(self, states, wide_goals, narrow_goals, rewards, values, log_probs, entropies, step_count=0, balls_snapshots=None, wide_only=False, narrow_only=False, wide_idxs=None, ppo_epochs=1, clip_eps=0.2, entropy_coef_override=None):
        """Update Manager policy with wide + narrow entropy regularization."""
        _entropy_coef = self.entropy_coef if entropy_coef_override is None else entropy_coef_override
        if len(rewards) == 0:
            return
        
        # Skip single-sample batches (can't normalize advantages)
        if len(rewards) <= 1:
            return

        verbose = self.action_verbose and (
            (self.diagnostic_checkstart and step_count < 15) or
            (step_count % self.diagnostic_interval == 0)
        )

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

        # Normalize advantages — guard against near-zero std (manager converged)
        raw_advantages = advantages.clone()
        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        advantages = advantages.clamp(-3.0, 3.0)

        old_log_probs_t = log_probs_tensor.detach()  # frozen reference for PPO ratio

        if ppo_epochs <= 1:
            # ── A2C: single gradient step (original behaviour) ──
            policy_loss = -(advantages.detach() * log_probs_tensor).mean()
            value_loss  = F.mse_loss(values_tensor, returns)

            if verbose:
                print(f"  Advantages (raw): {raw_advantages[:3].tolist()}")
                print(f"  Advantages (normalized): {advantages[:3].tolist()}")
                print(f"  Policy loss: {policy_loss.item():.4f}")

            passed_entropies_tensor = torch.stack(entropies)
            if narrow_only:
                narrow_entropy_mean = passed_entropies_tensor.mean()
                wide_entropy_mean   = torch.tensor(0.0, device=self.device)
            elif wide_only:
                wide_entropy_mean   = passed_entropies_tensor.mean()
                narrow_entropy_mean = torch.tensor(0.0, device=self.device)
            else:
                wide_entropy_mean = passed_entropies_tensor.mean()
                narrow_entropies = []
                balls_iter = balls_snapshots if balls_snapshots is not None else [None] * len(states)
                for wide_goal_i, balls_i in zip(wide_goals, balls_iter):
                    if balls_i:
                        nearest = min(balls_i, key=lambda b: abs(b[0]-wide_goal_i[0]) + abs(b[1]-wide_goal_i[1]))
                        dx_i = float(nearest[0] - wide_goal_i[0])
                        dy_i = float(nearest[1] - wide_goal_i[1])
                    else:
                        dx_i, dy_i = 0.0, 0.0
                    narrow_logits = self.narrow_head(torch.tensor([dx_i, dy_i], dtype=torch.float32, device=self.device))
                    if self._valid_cells is not None:
                        neighborhood_full_i = self.get_neighborhood(wide_goal_i)
                        valid_idx = [j for j, c in enumerate(neighborhood_full_i) if c in self._valid_cells]
                        if valid_idx:
                            idx_t = torch.tensor(valid_idx, dtype=torch.long, device=self.device)
                            narrow_logits = narrow_logits[idx_t]
                    narrow_probs = F.softmax(narrow_logits, dim=0)
                    narrow_entropy = -(narrow_probs * torch.log(narrow_probs + 1e-8)).sum()
                    narrow_entropies.append(narrow_entropy)
                narrow_entropy_mean = torch.stack(narrow_entropies).mean()

            if verbose:
                print(f"\nLoss components:")
                print(f"  Policy loss: {policy_loss.item():.6f}")
                print(f"  Value loss: {value_loss.item():.6f}")
                print(f"  Wide entropy: {wide_entropy_mean.item():.6f}")
                print(f"  Narrow entropy: {narrow_entropy_mean.item():.6f}")
                print(f"  Entropy coef: {_entropy_coef}")

            total_loss = (policy_loss +
                          self.value_coef * value_loss -
                          _entropy_coef * (wide_entropy_mean + narrow_entropy_mean))

            if verbose:
                print(f"  Total loss: {total_loss.item():.6f}")

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

        else:
            # ── PPO: K epochs, each replays the full LSTM sequence from scratch ──
            saved_hidden  = self.hidden_state
            saved_prev_gw = self.prev_wide_goal
            balls_iter    = balls_snapshots if balls_snapshots is not None else [None] * len(states)

            for _epoch in range(ppo_epochs):
                self.hidden_state   = None
                self.prev_wide_goal = (0, 0)

                new_lp, new_vals, new_ents = [], [], []
                for t, (s_t, gw_t, balls_t) in enumerate(zip(states, wide_goals, balls_iter)):
                    if t > 0:
                        self.prev_wide_goal = wide_goals[t - 1]

                    if narrow_only:
                        _, _, val_t = self.forward(s_t, balls_t)
                        if self.hidden_state:
                            self.hidden_state = tuple(h.detach() for h in self.hidden_state)
                        if balls_t:
                            nearest = min(balls_t, key=lambda b: abs(b[0]-gw_t[0]) + abs(b[1]-gw_t[1]))
                            dx_t = float(nearest[0] - gw_t[0])
                            dy_t = float(nearest[1] - gw_t[1])
                        else:
                            dx_t, dy_t = 0.0, 0.0
                        narrow_logits_t   = self.narrow_head(torch.tensor([dx_t, dy_t], dtype=torch.float32, device=self.device))
                        neighborhood_full_t = self.get_neighborhood(gw_t)
                        valid_idx_t       = [i for i, c in enumerate(neighborhood_full_t)
                                             if self._valid_cells is None or c in self._valid_cells]
                        if valid_idx_t:
                            idx_tensor    = torch.tensor(valid_idx_t, dtype=torch.long, device=self.device)
                            valid_probs_t = F.softmax(narrow_logits_t[idx_tensor], dim=0)
                            ng_t          = narrow_goals[t]
                            global_i      = neighborhood_full_t.index(ng_t) if ng_t in neighborhood_full_t else None
                            local_i       = valid_idx_t.index(global_i) if global_i is not None and global_i in valid_idx_t else None
                            lp_t  = torch.log(valid_probs_t[local_i] + 1e-8) if local_i is not None \
                                    else torch.tensor(-10.0, device=self.device)
                            ent_t = -(valid_probs_t * torch.log(valid_probs_t + 1e-8)).sum()
                        else:
                            lp_t  = torch.tensor(-10.0, device=self.device)
                            ent_t = torch.tensor(0.0,   device=self.device)
                    else:
                        wide_logits_t, _, val_t = self.forward(s_t, balls_t)
                        if self.hidden_state:
                            self.hidden_state = tuple(h.detach() for h in self.hidden_state)
                        probs_t = F.softmax(wide_logits_t, dim=0)
                        idx_t   = wide_idxs[t] if wide_idxs is not None else 0
                        lp_t    = torch.log(probs_t[idx_t] + 1e-8)
                        ent_t   = -(probs_t * torch.log(probs_t + 1e-8)).sum()

                    new_lp.append(lp_t)
                    new_vals.append(val_t.squeeze())
                    new_ents.append(ent_t.detach())

                new_lp_t   = torch.stack(new_lp)
                new_vals_t = torch.stack(new_vals)

                ratios     = torch.exp(new_lp_t - old_log_probs_t)
                surr1      = ratios * advantages.detach()
                surr2      = ratios.clamp(1 - clip_eps, 1 + clip_eps) * advantages.detach()
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(new_vals_t, returns)
                entropy_mean = torch.stack(new_ents).mean()

                total_loss = (policy_loss +
                              self.value_coef * value_loss -
                              _entropy_coef * entropy_mean)

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()

            self.hidden_state   = saved_hidden
            self.prev_wide_goal = saved_prev_gw
        
        if verbose:
            print(f"{'='*70}\n")

    def update_policy_batched(self, rollouts, ppo_epochs=4, clip_eps=0.2, entropy_coef_override=None):
        """
        Batched PPO update over multiple episode rollouts.
        Sequence-batches each episode's LSTM (one call per episode per epoch instead of T calls),
        then accumulates loss across all rollouts before a single backward pass per epoch.
        """
        _entropy_coef = self.entropy_coef if entropy_coef_override is None else entropy_coef_override

        # Pre-build fixed tensors for each rollout (done once, reused across epochs)
        prepared = []
        for (states, wide_goals, narrow_goals, rewards, values, log_probs, entropies,
             balls_snapshots, wide_idxs) in rollouts:
            if len(rewards) <= 1:
                continue

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            values_t  = torch.stack(values).squeeze().detach()
            if values_t.dim() == 0:
                values_t = values_t.unsqueeze(0)
            old_lps_t = torch.stack(log_probs).detach()

            R = 0; rets = []
            for r in reversed(rewards): R = r + self.gamma * R; rets.insert(0, R)
            returns_t = torch.tensor(rets, dtype=torch.float32, device=self.device)

            gae = 0; advantages = torch.zeros_like(rewards_t)
            for t in reversed(range(len(rewards))):
                next_val = values_t[t + 1] if t < len(rewards) - 1 else torch.tensor(0.0, device=self.device)
                delta    = rewards_t[t] + self.gamma * next_val - values_t[t]
                gae      = delta + self.gamma * 0.95 * gae
                advantages[t] = gae
            adv_std = advantages.std()
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            else:
                advantages = advantages - advantages.mean()
            advantages = advantages.clamp(-3.0, 3.0)

            # Build [1, T, 7] sequence input
            T_len = len(states)
            seq   = torch.zeros(T_len, 7, dtype=torch.float32, device=self.device)
            balls_iter = balls_snapshots if balls_snapshots is not None else [None] * T_len
            for t, (s_t, balls_t) in enumerate(zip(states, balls_iter)):
                prev_gw = wide_goals[t - 1] if t > 0 else (0, 0)
                if balls_t and len(balls_t) > 0:
                    nb = min(balls_t, key=lambda b: abs(b[0] - s_t[0]) + abs(b[1] - s_t[1]))
                    nb_x, nb_y, n_rem = float(nb[0]), float(nb[1]), float(len(balls_t))
                else:
                    nb_x, nb_y, n_rem = 0.0, 0.0, 0.0
                seq[t] = torch.tensor([s_t[0], s_t[1], prev_gw[0], prev_gw[1], nb_x, nb_y, n_rem])
            seq = seq.unsqueeze(0)  # [1, T, 7]

            wide_idxs_t = torch.tensor(wide_idxs, dtype=torch.long, device=self.device)
            prepared.append((seq, wide_idxs_t, old_lps_t, advantages, returns_t))

        if not prepared:
            return

        for _epoch in range(ppo_epochs):
            total_loss = torch.tensor(0.0, device=self.device)
            for seq, wide_idxs_t, old_lps_t, advantages, returns_t in prepared:
                lstm_out, _       = self.lstm(seq, None)              # [1, T, 64]
                feats             = lstm_out.squeeze(0)               # [T, 64]
                wide_logits_all   = self.wide_head(feats)             # [T, N_pivots]
                values_new        = self.critic(feats).squeeze(-1)    # [T]

                probs_all  = F.softmax(wide_logits_all, dim=-1)
                new_lps_t  = torch.log(probs_all.gather(1, wide_idxs_t.unsqueeze(1)).squeeze(1) + 1e-8)
                ents_t     = -(probs_all * torch.log(probs_all + 1e-8)).sum(dim=-1)

                ratios     = torch.exp(new_lps_t - old_lps_t)
                surr1      = ratios * advantages
                surr2      = ratios.clamp(1 - clip_eps, 1 + clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.mse_loss(values_new, returns_t)
                entropy_mean = ents_t.mean()

                total_loss = total_loss + (policy_loss + self.value_coef * value_loss
                                           - _entropy_coef * entropy_mean)

            total_loss = total_loss / len(prepared)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()


#----------------------------------------------------------------------------#
#                            WORKER FSM STATE ENUM                           #
#----------------------------------------------------------------------------#

class WorkerState(Enum):
    FINDING   = 1  # Navigating toward nearest pivotal state
    TRAVERSAL = 2  # Executing deterministic graph traversal to wide_goal
    NARROW_GOAL = 3  # MLP navigating from wide_goal to narrow_goal

#----------------------------------------------------------------------------#
#                            HIERARCHICAL WORKER                             #
#----------------------------------------------------------------------------#

class HierarchicalWorker(nn.Module):
    """
    Phase 3: Hierarchical Worker that executes Manager's goals.
    Paper: "Worker can leverage the graph to easily traverse to pivotal states"
    """
    
    def __init__(self,
                 world_graph,
                 pivotal_states: List[Tuple[int, int]],
                 lr: float = 5e-3,
                 verbose: bool = False,
                 goal_policy=None,
                 maze_size: int = 24,
                 neighborhood_size: int = 3,
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
        self.maze_size = maze_size
        self.neighborhood_size = neighborhood_size
        self.world_graph = world_graph
        self.pivotal_states = set(pivotal_states)

        # Phase 1 navigator — fine-tuned during Phase 3.
        # Stored via object.__setattr__ to prevent PyTorch from registering it as a
        # submodule: otherwise self.parameters() would include GCP params, causing
        # double-counting in the optimizer and polluting worker.state_dict().
        object.__setattr__(self, 'goal_policy', goal_policy)

        # A2C-MLP architecture (17 inputs: asymmetric agent-relative wall patch + goal_dx, goal_dy)
        # Patch: 5x5 square sliced, rotated (row-0=forward), cropped to (patch_rows_forward+1)x5.
        # Agent sits at last row; rows behind are discarded (worker has no backward action).
        self.wall_patch_size = 5        # square slice size; half=2 used for pre-padding
        self.patch_rows_forward = 2     # rows visible ahead (excl. agent row)
        self.wall_mask: np.ndarray = None  # pre-padded (height+4, width+4), built by build_wall_mask()
        _patch_inputs = (self.patch_rows_forward + 1) * self.wall_patch_size  # 3*5=15
        self.net = nn.Sequential(
            nn.Linear(_patch_inputs + 2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        ).to(device)

        self.actor = nn.Linear(64, 3).to(device)
        self.critic = nn.Linear(64, 1).to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Traversal state
        self.current_traversal_path = []  # Path of PIVOTAL states, e.g., [(A), (B), (C)]
        self.traversal_step = 0           # Index into self.current_traversal_path
        self.current_edge_actions = []    # Action sequence for ONE edge, e.g., [1, 1, 2, 2]
        self.current_action_idx = 0       # Index into self.current_edge_actions

        # FSM state — reset to FINDING on each new manager goal
        self._worker_state: WorkerState = WorkerState.FINDING
        self._traversal_starts_this_episode: int = 0
        self._last_local_goal = None           # set each MLP step; None for traversal steps
        self._finding_local_goal = None        # current local goal being chased in FINDING
        self._finding_steps: int = 0           # steps spent chasing _finding_local_goal
        self._finding_blacklist: set = set()   # pivots unreachable as local goals in FINDING
        self.finding_local_timeout: int = 30   # max steps per local goal in FINDING before skip
        self._narrow_goal_steps: int = 0       # steps spent in NARROW_GOAL since entry
        self.narrow_goal_timeout: int = 30     # steps in NARROW_GOAL without reaching it → force manager replanning
        self.narrow_goal_timed_out: bool = False  # trainer checks this to trigger goal change
        self._spinning_steps: int = 0          # consecutive NARROW_GOAL steps without x,y change
        self.spinning_timeout: int = 10        # spinning steps before forcing replanning

        # Set of traversable (x,y) cells — used by forward() to compute wall flags.
        # Must be populated before the first call to forward() (set by trainer/pretrain).
        self.valid_cells: set = set()

        # Hyperparameters
        self.gamma = 0.99
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        # Goal-delta normalization divisor — kept at 1.0 (raw cell deltas) throughout pretrain
        # and Phase 3.  GCP also uses raw deltas (aligned).  Default 1.0.
        self.goal_norm_div: float = 1.0
    
    def reset_worker_state(self):
        """Reset traversal state and FSM back to FINDING."""
        self.current_traversal_path = []
        self.traversal_step = 0
        self.current_edge_actions = []
        self.current_action_idx = 0
        self._worker_state = WorkerState.FINDING
        self._finding_local_goal = None
        self._finding_steps = 0
        self._finding_blacklist = set()
        self._narrow_goal_steps = 0
        self.narrow_goal_timed_out = False
        self._spinning_steps = 0

    def report_step(self, prev_state: Tuple[int, int], next_state: Tuple[int, int]):
        """Called by trainer after each env step. Detects spinning in NARROW_GOAL and fires replanning."""
        if self._worker_state != WorkerState.NARROW_GOAL:
            self._spinning_steps = 0
            return
        if prev_state == next_state:
            self._spinning_steps += 1
            if self._spinning_steps >= self.spinning_timeout:
                self.narrow_goal_timed_out = True
        else:
            self._spinning_steps = 0

    def build_wall_mask(self, width: int, height: int):
        """Precompute pre-padded wall mask from valid_cells. Call once after setting valid_cells."""
        half = self.wall_patch_size // 2
        raw = np.ones((height, width), dtype=np.float32)
        for (x, y) in self.valid_cells:
            raw[y, x] = 0.0
        self.wall_mask = np.pad(raw, half, mode='constant', constant_values=1.0)

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
    
    def forward(self, state: Tuple[int, int], agent_dir: int, narrow_goal: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Worker network.

        Args:
            state: Current agent position
            agent_dir: Current agent facing direction (0=right,1=down,2=left,3=up)
            narrow_goal: Manager's narrow goal

        Returns:
            action_logits: Logits over 3 navigation actions
            value: State value estimate
        """
        # Asymmetric agent-relative wall patch: slice 5x5, rotate (row-0=forward), crop behind.
        # k=(dir+1)%4 CCW: dir0(east)→k1, dir1(south)→k2, dir2(west)→k3, dir3(north)→k0
        # After crop: shape (patch_rows_forward+1, 5) = (3,5). Agent at last row, center col.
        ax, ay = state
        patch = self.wall_mask[ay : ay + self.wall_patch_size, ax : ax + self.wall_patch_size]
        patch = np.rot90(patch, k=(agent_dir + 1) % 4)
        patch = patch[:self.patch_rows_forward + 1, :]
        # Goal in agent-relative frame (CW rotation by agent_dir*90°):
        # dir0: fwd=+x, rgt=+y | dir1: fwd=+y, rgt=-x | dir2: fwd=-x, rgt=-y | dir3: fwd=-y, rgt=+x
        wx = float(narrow_goal[0] - ax) / self.goal_norm_div
        wy = float(narrow_goal[1] - ay) / self.goal_norm_div
        if   agent_dir == 0: goal_fwd, goal_rgt =  wx,  wy
        elif agent_dir == 1: goal_fwd, goal_rgt =  wy, -wx
        elif agent_dir == 2: goal_fwd, goal_rgt = -wx, -wy
        else:                goal_fwd, goal_rgt = -wy,  wx
        net_input = torch.tensor(
            patch.flatten().tolist() + [goal_fwd, goal_rgt],
            dtype=torch.float32, device=self.device,
        )  # [17]
        features = self.net(net_input)               # [64]
        
        # Action and value
        action_logits = self.actor(features)  # [3]
        value = self.critic(features)  # [1]
        
        return action_logits, value
    
    def get_action(self, state: Tuple[int, int], wide_goal: Tuple[int, int], narrow_goal: Tuple[int, int], agent_dir: int):
        """Worker selects action via 3-state FSM: FINDING → TRAVERSAL → NARROW_GOAL."""
        self._last_local_goal = None  # cleared every step; only set for MLP steps

        # ── FSM TRANSITIONS ────────────────────────────────────────────────────
        if self._worker_state == WorkerState.FINDING:
            if state == wide_goal:
                # Already at wide_goal — enter NARROW_GOAL directly
                if diag: print(f"\n[WORKER] FINDING: already at wide_goal {state}. →NARROW_GOAL")
                self._worker_state = WorkerState.NARROW_GOAL
            elif state in self.pivotal_states and state not in self._finding_blacklist:
                path = self.plan_traversal(state, wide_goal)
                if path:
                    self._finding_blacklist.clear()
                    self._worker_state = WorkerState.TRAVERSAL
                    self.current_traversal_path = path
                    self.traversal_step = 0
                    self._traversal_starts_this_episode += 1
                    if diag:
                        print(f"\n[WORKER] FINDING→TRAVERSAL at {state}, target gw={wide_goal}")
                        print(f"  Path: {self.current_traversal_path}")

        # ── STATE EXECUTION ─────────────────────────────────────────────────────
        if self._worker_state == WorkerState.TRAVERSAL:
            if not self.current_edge_actions:
                if self.traversal_step < len(self.current_traversal_path) - 1:
                    start_node = self.current_traversal_path[self.traversal_step]
                    end_node   = self.current_traversal_path[self.traversal_step + 1]

                    if state != start_node:
                        if diag:
                            print(f"  [WORKER] DESYNC: at {state}, expected {start_node}. →FINDING")
                        self.reset_worker_state()   # back to FINDING
                    else:
                        coord_path = self.world_graph.get_edge_path(start_node, end_node)
                        if coord_path:
                            self.current_edge_actions = self.generate_actions_from_path(coord_path, agent_dir)
                            self.current_action_idx = 0
                            if diag:
                                print(f"  [WORKER] Loading edge {start_node}→{end_node} "
                                      f"({len(self.current_edge_actions)} actions)")
                        else:
                            if diag: print(f"  [WORKER] Edge {start_node}→{end_node} missing. →FINDING")
                            self.reset_worker_state()   # back to FINDING
                else:
                    # All edges done — switch to narrow-goal navigation
                    if diag: print("  [WORKER] Traversal complete. →NARROW_GOAL")
                    self.reset_worker_state()               # clears path/edge buffers + sets FINDING
                    self._worker_state = WorkerState.NARROW_GOAL

            if self.current_edge_actions and self.current_action_idx < len(self.current_edge_actions):
                action = self.current_edge_actions[self.current_action_idx]
                self.current_action_idx += 1
                if self.current_action_idx >= len(self.current_edge_actions):
                    self.current_edge_actions = []
                    self.traversal_step += 1
                    if diag: print(f"  [WORKER] Edge done. traversal_step={self.traversal_step}")
                with torch.no_grad():
                    _, value = self.forward(state, agent_dir, narrow_goal)
                return action, torch.tensor(-1.0, device=self.device), value.squeeze()

        # ── MLP NAVIGATION (FINDING or NARROW_GOAL) ────────────────────────────
        if self._worker_state == WorkerState.FINDING:
            # Steer toward nearest pivotal state; exclude current pos and timed-out pivots.
            # Always prefer wide_goal directly when it's a valid candidate — avoids the
            # one-step display artifact where the agent lands on wide_goal while targeting
            # a different pivot, and makes FINDING take the most direct route.
            candidates = [p for p in self.pivotal_states
                          if p != state and p not in self._finding_blacklist]
            if not candidates:
                if diag: print("  [WORKER] All pivots blacklisted → clearing blacklist")
                self._finding_blacklist.clear()
                self._finding_local_goal = None
                self._finding_steps = 0
                candidates = [p for p in self.pivotal_states if p != state]
            if wide_goal in candidates:
                local_goal = wide_goal
            else:
                local_goal = (min(candidates, key=lambda p: abs(p[0] - state[0]) + abs(p[1] - state[1]))
                              if candidates else narrow_goal)
            # Timer: if same target, tick; on timeout blacklist it and pick next
            if local_goal == self._finding_local_goal:
                self._finding_steps += 1
                if self._finding_steps >= self.finding_local_timeout:
                    if diag: print(f"  [WORKER] FINDING local timeout on {local_goal} → blacklist")
                    self._finding_blacklist.add(local_goal)
                    self._finding_steps = 0
                    self._finding_local_goal = None
                    candidates = [p for p in self.pivotal_states
                                  if p != state and p not in self._finding_blacklist]
                    local_goal = (min(candidates, key=lambda p: abs(p[0] - state[0]) + abs(p[1] - state[1]))
                                  if candidates else narrow_goal)
                    self._finding_local_goal = local_goal
            else:
                self._finding_local_goal = local_goal
                self._finding_steps = 0
        else:   # NARROW_GOAL
            if state != narrow_goal:  # don't count steps once already on the goal cell
                self._narrow_goal_steps += 1
                if self._narrow_goal_steps >= self.narrow_goal_timeout:
                    if diag: print(f"  [WORKER] NARROW_GOAL timeout ({self._narrow_goal_steps} steps) → request replanning")
                    self.narrow_goal_timed_out = True
            local_goal = narrow_goal

        self._last_local_goal = local_goal  # expose for diagnostics

        action_logits, value = self.forward(state, agent_dir, local_goal)
        probs = F.softmax(action_logits, dim=0)
        dist  = torch.distributions.Categorical(probs)
        idx   = dist.sample()
        return idx.item(), dist.log_prob(idx), value.squeeze()

    def compute_reward(self, prev_state: Tuple[int, int], current_state: Tuple[int, int], wide_goal: Tuple[int, int], narrow_goal: Tuple[int, int]) -> float:
        """
        Compute Worker's reward.
        Paper: "Worker receives rewards from Manager by reaching subgoals"
        """
        if current_state == narrow_goal:
            return 1.0
        prev_dist = abs(prev_state[0] - narrow_goal[0]) + abs(prev_state[1] - narrow_goal[1])
        curr_dist = abs(current_state[0] - narrow_goal[0]) + abs(current_state[1] - narrow_goal[1])
        return (prev_dist - curr_dist) * 0.1 - 0.01
    
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
            if hasattr(goal_policy, 'net'):
                # First layer: GCP input_size=6, Worker input_size=9.
                # Copy the 6 base columns; zero-init the 3 wall-flag columns so they
                # start silent and the pre-trained navigation features are preserved.
                src0 = goal_policy.net[0]
                if hasattr(src0, 'weight') and src0.weight.shape[1] == 6:
                    self.net[0].weight[:, :6].copy_(src0.weight)
                    nn.init.zeros_(self.net[0].weight[:, 6:])
                    self.net[0].bias.copy_(src0.bias)
                # Second layer: [64,64] unchanged — direct copy
                src2 = goal_policy.net[2]
                if hasattr(src2, 'weight') and src2.weight.shape == self.net[2].weight.shape:
                    self.net[2].weight.copy_(src2.weight)
                    self.net[2].bias.copy_(src2.bias)

            # Copy actor (first 3 of GCP's 7 actions)
            if hasattr(goal_policy, 'actor'):
                self.actor.weight.copy_(goal_policy.actor.weight[:3, :])
                self.actor.bias.copy_(goal_policy.actor.bias[:3])

            # Copy critic
            if hasattr(goal_policy, 'critic'):
                self.critic.weight.copy_(goal_policy.critic.weight)
                self.critic.bias.copy_(goal_policy.critic.bias)

        print("Worker initialized from goal policy (9-dim MLP: state+dir+goal+walls)")


    def update_policy(self, states: List, actions: List, rewards: List,
                     values: List[torch.Tensor], log_probs: List[torch.Tensor],
                     next_states=None, dones=None,
                     ppo_epochs: int = 4, clip_eps: float = 0.2,
                     gae_lambda: float = 0.95) -> dict:
        """
        Update Worker policy with PPO (when next_states provided) or MC A2C (fallback).
        PPO: GAE advantages + clipped surrogate + K epochs.
        A2C fallback: MC returns, single epoch — used by Phase 3 HierarchicalTrainer.
        """
        if len(rewards) == 0:
            return {}

        T = len(rewards)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values_t  = torch.stack(values).squeeze()
        if values_t.dim() == 0:
            values_t = values_t.unsqueeze(0)

        old_log_probs = torch.stack(log_probs).detach()

        if next_states is not None:
            # GAE advantages
            dones_t = torch.tensor(
                [float(d) for d in dones], dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                next_vals = []
                for (ns, nad, ng) in next_states:
                    _, nv = self.forward(ns, nad, ng)
                    next_vals.append(nv.squeeze())
                next_values_t = torch.stack(next_vals)

            advantages = torch.zeros(T, device=self.device)
            gae = 0.0
            for t in reversed(range(T)):
                delta = (rewards_t[t]
                         + self.gamma * next_values_t[t] * (1 - dones_t[t])
                         - values_t[t].detach())
                gae = delta + self.gamma * gae_lambda * (1 - dones_t[t]) * gae
                advantages[t] = gae
            returns = advantages + values_t.detach()
            n_epochs = ppo_epochs
        else:
            # MC fallback for Phase 3 (no next_states tracked)
            mc = []
            R = 0.0
            for r in reversed(rewards):
                R = r + self.gamma * R
                mc.insert(0, R)
            returns = torch.tensor(mc, dtype=torch.float32, device=self.device)
            advantages = returns - values_t.detach()
            n_epochs = 1

        adv_mean = advantages.mean().item()
        adv_std  = advantages.std().item() if T > 1 else 0.0
        if T > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        params = list(self.parameters())

        last_metrics: dict = {}
        for _ in range(n_epochs):
            new_log_probs_list, new_values_list, entropy_list = [], [], []
            for i, (s, ad, ng) in enumerate(states):
                action_logits, new_val = self.forward(s, ad, ng)
                dist = torch.distributions.Categorical(logits=action_logits)
                new_log_probs_list.append(
                    dist.log_prob(torch.tensor(actions[i], device=self.device))
                )
                new_values_list.append(new_val.squeeze())
                entropy_list.append(dist.entropy())

            new_log_probs_t = torch.stack(new_log_probs_list)
            new_values_t    = torch.stack(new_values_list)
            entropy         = torch.stack(entropy_list).mean()

            ratios      = torch.exp(new_log_probs_t - old_log_probs)
            surr1       = ratios * advantages.detach()
            surr2       = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages.detach()
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(new_values_t, returns)
            total_loss  = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=0.5).item()
            self.optimizer.step()

            last_metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss':  value_loss.item(),
                'entropy':     entropy.item(),
                'grad_norm':   grad_norm,
            }

        return {
            **last_metrics,
            'adv_mean':    adv_mean,
            'adv_std':     adv_std,
            'value_mean':  values_t.mean().item(),
            'return_mean': returns.mean().item(),
        }

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


#----------------------------------------------------------------------------#
#                            HIERARCHICAL TRAINER                            #
#----------------------------------------------------------------------------#

class HierarchicalTrainer:
    def __init__(self, manager: HierarchicalManager, worker: HierarchicalWorker,
                 env, horizon: int = 15,
                 diagnostic_interval: int = 30,
                 diagnostic_checkstart: bool = True,
                 workershaping=True,
                 managershaping=True,
                 narrow_shaping_weight: float = 1.0,
                 traversal_shaping_weight: float = 2.0,
                 goal_timeout: int = 3,
                 instant_traversal: bool = False):
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
            'worker_local_goal_achievement': [],
            'balls_collected_per_episode': [],
            'manager_rewards_mean': [],
            'manager_rewards_std': [],
            'goal_distance_to_balls': [],
            'manager_value_mean': [],
            'worker_value_mean': [],
            'episode_rewards': []
        }

        self.worker_shaping_weight=0.2   # max ~0.15/horizon << success reward 1.0
        self.manager_shaping_weight=5
        self.narrow_shaping_weight=narrow_shaping_weight
        self.traversal_shaping_weight=traversal_shaping_weight
        self.manhattan_distance_rew_shaping=workershaping
        self.manager_reward_shaping=managershaping
        self.instant_traversal = instant_traversal
    
    def train_episode(self, max_steps: int = 200, full_breakdown_every=1):
        """Train one episode with comprehensive diagnostics."""

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
        self.worker.valid_cells = valid_cells

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
        steps_on_goal = 0
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
        local_goal_steps_total = 0
        local_goal_steps_hit   = 0
        
        # Manager experience accumulation
        manager_states = []
        manager_wide_goals = []
        manager_narrow_goals = []
        manager_rewards = []
        manager_values = []
        manager_log_probs = []
        manager_entropies_for_update = []
        manager_balls_snapshots = []
        
        horizon_counter = 0
        wide_blacklist: set = set()   # wide goals to avoid after narrow_goal timeout; cleared on goal reached
        narrow_blacklist: set = set() # narrow goals already reached without collecting a ball this episode

        while episode_steps < max_steps:
            # ── GOAL SELECTION (persistence) ──────────────────────────────────
            need_new_goal = (
                active_wide_goal is None
                or goal_reached_prev
                or ball_collected_prev
                or steps_on_goal >= self.goal_timeout
                or self.worker.narrow_goal_timed_out
            )

            if need_new_goal:
                if active_narrow_goal is not None:
                    narrow_blacklist.add(active_narrow_goal)  # always blacklist: narrow goals never re-selected in same episode
                if ball_collected_prev or goal_reached_prev:
                    wide_blacklist.clear()
                elif self.worker.narrow_goal_timed_out:
                    # Only blacklist the pivot when the goal was NOT reached (worker was genuinely stuck)
                    if active_wide_goal is not None:
                        wide_blacklist.add(active_wide_goal)
                if active_wide_goal is not None:
                    self.worker.reset_worker_state()  # resets traversal buffers + FSM to FINDING

                no_repeat_blacklist = wide_blacklist | ({active_wide_goal} if active_wide_goal is not None else set())
                wide_goal, narrow_goal, manager_log_prob, manager_value, entropy = self.manager.get_manager_action(
                    state, step_count=self.global_step_counter, valid_cells=valid_cells,
                    active_balls=list(self.env.active_balls), wide_blacklist=no_repeat_blacklist,
                    narrow_blacklist=narrow_blacklist
                )
                if self.manager.hidden_state is not None:
                    self.manager.hidden_state = tuple(h.detach() for h in self.manager.hidden_state)

                active_wide_goal = wide_goal
                active_narrow_goal = narrow_goal
                active_log_prob = manager_log_prob
                active_value = manager_value
                active_entropy = entropy.detach()
                steps_on_goal = 0
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
            traversal_completed_this_horizon = False

            # Save starting state for Manager reward shaping
            starting_state_snapshot = state
            starting_balls_snapshot = list(self.env.active_balls)

            for h in range(self.horizon):
                # BEFORE taking action, record distance FOR SHAPING
                old_dist_narrow = manhattan_distance(state, narrow_goal)
                old_dist_wide = manhattan_distance(state, wide_goal)

                # Detect traversal completion: traversal was active but will finish inside get_action
                was_traversing = bool(self.worker.current_traversal_path)

                # Capture direction before step so forward() and worker_states stay in sync
                agent_dir = self.env.agent_dir

                # Worker selects action
                action, worker_log_prob, worker_value = self.worker.get_action(
                    state, wide_goal, narrow_goal,
                    agent_dir=agent_dir
                )

                # Instant traversal: teleport to wide_goal instead of executing graph actions
                if self.instant_traversal and self.worker._worker_state == WorkerState.TRAVERSAL:
                    self.env.agent_pos = np.array(wide_goal)
                    state = wide_goal
                    self.worker.reset_worker_state()
                    self.worker._worker_state = WorkerState.NARROW_GOAL
                    traversal_completed_this_horizon = True
                    continue

                if was_traversing and not self.worker.current_traversal_path and state == wide_goal:
                    traversal_completed_this_horizon = True
                
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

                self.worker.report_step(state, next_state)

                # Local goal achievement tracking (MLP steps only)
                if self.worker._last_local_goal is not None:
                    local_goal_steps_total += 1
                    if next_state == self.worker._last_local_goal:
                        local_goal_steps_hit += 1

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
                worker_reward = self.worker.compute_reward(state, next_state, wide_goal, narrow_goal)
                # Add Manhattan Shaping

                if self.manhattan_distance_rew_shaping:
                    worker_reward += progress_bonus * self.worker_shaping_weight
                
                # Store Worker experience
                # Only store if worker is NOT traversing
                if not self.worker.current_traversal_path:
                    worker_states.append((state, agent_dir, narrow_goal))
                    worker_actions.append(action)
                    worker_rewards.append(worker_reward)
                    worker_values.append(worker_value)
                    worker_log_probs.append(worker_log_prob)

                # Track episode stats
                horizon_env_reward += env_reward
                episode_reward += env_reward

                self.global_step_counter += 1
                episode_steps += 1
                state = next_state

                # TEMP DIAGNOSTIC — remove once bug is identified
                _dist_to_narrow = abs(next_state[0] - narrow_goal[0]) + abs(next_state[1] - narrow_goal[1])
                if _dist_to_narrow <= 1:
                    print(f"[NARROW DBG] dist={_dist_to_narrow} "
                          f"next_state={next_state} ({type(next_state[0]).__name__}) "
                          f"narrow_goal={narrow_goal} ({type(narrow_goal[0]).__name__}) "
                          f"equal={next_state == narrow_goal} "
                          f"worker_state={self.worker._worker_state.name}")

                if next_state == narrow_goal:
                    goal_reached_this_horizon = True
                    break  # don't waste remaining horizon steps after goal reached

                if terminated or truncated:
                    break
            
            
            # Update Worker only when goal was reached — preserves pretrained weights on failures
            if len(worker_rewards) > 0 and goal_reached_this_horizon:
                self.worker.update_policy(
                    worker_states,
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
                        maze_diagonal = (self.env.width - 2) + (self.env.height - 2)
                        progress_reward = progress * self.manager_shaping_weight / maze_diagonal
                        manager_reward += progress_reward

            # Bonus for choosing a narrow goal close to a ball (only proximity, no base)
            if goal_reached_this_horizon and len(starting_balls_snapshot) > 0:
                dist_narrow_to_ball = min(
                    manhattan_distance(narrow_goal, ball) for ball in starting_balls_snapshot
                )
                manager_reward += 0.5 / (1.0 + dist_narrow_to_ball) ** 2.5

            # Traversal completion bonus: reward manager for arriving at wide_goal via graph,
            # scaled by proximity of wide_goal to nearest ball. Fires reliably (traversal is
            # deterministic) giving the wide head a direct gradient toward ball-adjacent pivotals.
            # Cutoff at neighborhood_size: if the pivot is further than r from every ball,
            # the narrow head cannot reach that ball anyway — rewarding it is misleading.
            if traversal_completed_this_horizon and len(starting_balls_snapshot) > 0:
                dist_wide_to_ball = min(
                    manhattan_distance(wide_goal, ball) for ball in starting_balls_snapshot
                )
                if dist_wide_to_ball <= self.manager.neighborhood_size:
                    manager_reward += self.traversal_shaping_weight / (1.0 + dist_wide_to_ball) ** 2

            # Push manager experience every horizon
            manager_states.append(starting_state_snapshot)
            manager_wide_goals.append(active_wide_goal)
            manager_narrow_goals.append(active_narrow_goal)
            manager_rewards.append(manager_reward)
            manager_values.append(active_value)
            manager_log_probs.append(active_log_prob)
            manager_entropies_for_update.append(active_entropy)
            manager_balls_snapshots.append(starting_balls_snapshot)

            steps_on_goal += self.horizon
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
                step_count=self.global_step_counter,
                balls_snapshots=manager_balls_snapshots
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
        local_goal_rate = local_goal_steps_hit / local_goal_steps_total if local_goal_steps_total > 0 else 0
        
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
        self.diagnostic_history['worker_local_goal_achievement'].append(local_goal_rate)
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
            print(f"  Local goal achievement rate: {local_goal_rate*100:.1f}%  ({local_goal_steps_hit}/{local_goal_steps_total} MLP steps)")
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
