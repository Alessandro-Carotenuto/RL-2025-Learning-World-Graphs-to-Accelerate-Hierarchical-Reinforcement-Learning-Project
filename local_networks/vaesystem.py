import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional

import numpy as np

from local_distributions.hardkuma import HardKumaraswamy, BetaDistribution


class PriorNetwork(nn.Module):
    """
    Prior network that learns state-conditioned Beta distribution parameters.
    Paper: pψ(zt|st) = Beta(αt, βt)
    """
    
    def __init__(self, state_dim, hidden_dim=64):
        """
        Args:
            state_dim (int): Dimension of state representation
            hidden_dim (int): Hidden dimension for BiLSTM
        """
        super().__init__()
        
        # Bidirectional LSTM as specified in paper
        self.bilstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layers for Beta parameters (alpha, beta)
        # BiLSTM outputs 2*hidden_dim due to bidirectional
        self.alpha_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive alpha
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive beta
        )
    
    def forward(self, states):
        """
        Forward pass to get Beta parameters.
        
        Args:
            states (torch.Tensor): State sequence [batch_size, seq_len, state_dim]
            
        Returns:
            tuple: (alpha, beta) parameters for Beta distribution
                   Each has shape [batch_size, seq_len]
        """
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(states)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Get Beta parameters
        alpha = self.alpha_head(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        beta = self.beta_head(lstm_out).squeeze(-1)    # [batch_size, seq_len]
        
        return alpha, beta


class InferenceNetwork(nn.Module):
    """
    Inference network (encoder) that outputs HardKuma parameters from state-action pairs.
    Paper: qφ(zt|at,st) = HardKuma(α̃t, 1)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Args:
            state_dim (int): Dimension of state representation
            action_dim (int): Dimension of action representation (usually 1 for discrete)
            hidden_dim (int): Hidden dimension for BiLSTM
        """
        super().__init__()
        
        # Bidirectional LSTM processing state-action pairs
        self.bilstm = nn.LSTM(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer for HardKuma alpha parameter
        self.alpha_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive alpha
        )
    
    def forward(self, states, actions):
        """
        Forward pass to get HardKuma parameters.
        
        Args:
            states (torch.Tensor): State sequence [batch_size, seq_len, state_dim]
            actions (torch.Tensor): Action sequence [batch_size, seq_len, action_dim]
            
        Returns:
            torch.Tensor: Alpha parameters for HardKuma [batch_size, seq_len]
        """
        # Concatenate states and actions
        state_action = torch.cat([states, actions], dim=-1)  # [batch_size, seq_len, state_dim + action_dim]
        
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(state_action)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Get HardKuma alpha parameter
        alpha = self.alpha_head(lstm_out).squeeze(-1)  # [batch_size, seq_len]
        
        return alpha


class GenerationNetwork(nn.Module):
    """
    Generation network (decoder) that reconstructs actions from masked states.
    Paper: Reconstructs {at}^(T-1)_0 using only states where zt = 1
    """
    
    def __init__(self, state_dim, action_vocab_size, hidden_dim=64):
        """
        Args:
            state_dim (int): Dimension of state representation
            action_vocab_size (int): Number of possible actions (e.g., 7 for MiniGrid)
            hidden_dim (int): Hidden dimension for BiLSTM
        """
        super().__init__()
        
        # Bidirectional LSTM for processing masked states
        self.bilstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layer for action predictions
        self.action_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_vocab_size)
            # No activation - raw logits for CrossEntropyLoss
        )
    
    def forward(self, masked_states):
        """
        Forward pass to reconstruct actions.
        
        Args:
            masked_states (torch.Tensor): States where zt=1 [batch_size, masked_seq_len, state_dim]
            
        Returns:
            torch.Tensor: Action logits [batch_size, masked_seq_len, action_vocab_size]
        """
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(masked_states)  # [batch_size, masked_seq_len, 2*hidden_dim]
        
        # Get action predictions
        action_logits = self.action_head(lstm_out)  # [batch_size, masked_seq_len, action_vocab_size]
        
        return action_logits


class StateEncoder(nn.Module):
    """
    Helper class to encode different state representations into fixed-dimension vectors.
    Handles various MiniGrid state formats.
    """
    
    def __init__(self, state_format="coordinates", grid_size=24, output_dim=32):
        """
        Args:
            state_format (str): "coordinates", "grid_obs", or "flattened"
            grid_size (int): Size of the grid (for coordinate normalization)
            output_dim (int): Output dimension for encoded states
        """
        super().__init__()
        
        self.state_format = state_format
        self.grid_size = grid_size
        
        if state_format == "coordinates":
            # Simple coordinate encoding (x, y) -> embedding
            self.encoder = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim)
            )
        elif state_format == "grid_obs":
            # Grid observation encoding (assumes flattened grid)
            self.encoder = nn.Sequential(
                nn.Linear(grid_size * grid_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        else:  # flattened or custom
            # Generic linear encoder
            self.encoder = nn.Sequential(
                nn.Linear(64, 32),  # Adjust input size as needed
                nn.ReLU(),
                nn.Linear(32, output_dim)
            )
    
    def forward(self, states):
        """
        Encode states to fixed dimension.
        
        Args:
            states: Raw states in various formats
            
        Returns:
            torch.Tensor: Encoded states [batch_size, seq_len, output_dim]
        """
        if self.state_format == "coordinates":
            # Normalize coordinates to [0, 1]
            normalized = states.float() / self.grid_size
            return self.encoder(normalized)
        else:
            return self.encoder(states.float())


class ActionEncoder(nn.Module):
    """
    Helper class to encode discrete actions into continuous representations.
    """
    
    def __init__(self, num_actions=7, embedding_dim=8):
        """
        Args:
            num_actions (int): Number of possible actions
            embedding_dim (int): Dimension of action embeddings
        """
        super().__init__()
        
        # Embedding layer for discrete actions
        self.embedding = nn.Embedding(num_actions, embedding_dim)
    
    def forward(self, actions):
        """
        Encode discrete actions.
        
        Args:
            actions (torch.Tensor): Discrete action indices [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Action embeddings [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(actions.long())


class VAESystem(nn.Module):
    """
    Main VAE system that orchestrates pivotal state discovery.
    
    Combines all networks, implements ELBO loss with regularization,
    manages training loop, and extracts final pivotal states.
    """
    
    def __init__(
        self,
        state_dim: int = 32,
        action_vocab_size: int = 7,
        hidden_dim: int = 64,
        mu0: float = 5.0,  # Target number of pivotal states per trajectory
        grid_size: int = 24,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the VAE system.
        
        Args:
            state_dim: Dimension of encoded states
            action_vocab_size: Number of possible actions (MiniGrid = 7)
            hidden_dim: Hidden dimension for BiLSTM networks
            mu0: Target sparsity (expected number of kept states)
            grid_size: Size of the grid environment
            device: Computing device
        """
        super().__init__()  # Initialize nn.Module
        
        self.current_kl_weight = 0.0  # Add this
        
        self.device = device
        self.mu0 = mu0
        self.state_dim = state_dim
        self.action_vocab_size = action_vocab_size
        
        # Initialize encoders
        self.state_encoder = StateEncoder(
            state_format="coordinates", 
            grid_size=grid_size, 
            output_dim=state_dim
        ).to(device)
        
        self.action_encoder = ActionEncoder(
            num_actions=action_vocab_size,
            embedding_dim=8
        ).to(device)
        
        # Initialize VAE networks
        self.prior_net = PriorNetwork(state_dim, hidden_dim).to(device)
        self.inference_net = InferenceNetwork(state_dim, 8, hidden_dim).to(device)  # action_dim = 8
        self.generation_net = GenerationNetwork(state_dim, action_vocab_size, hidden_dim).to(device)
        
        # Lagrangian multipliers (learnable parameters for automatic weight tuning)
        self.lambda_kl = nn.Parameter(torch.tensor(1.0))      # KL divergence weight
        self.lambda_l0 = nn.Parameter(torch.tensor(1.0))      # L0 sparsity weight  
        self.lambda_lt = nn.Parameter(torch.tensor(1.0))      # Transition weight
        
        # Optimizers
        self.vae_optimizer = optim.Adam(
            list(self.state_encoder.parameters()) +
            list(self.action_encoder.parameters()) +
            list(self.prior_net.parameters()) +
            list(self.inference_net.parameters()) +
            list(self.generation_net.parameters()),
            lr=1e-3
        )
        
        self.lambda_optimizer = optim.Adam(
            [self.lambda_kl, self.lambda_l0, self.lambda_lt],
            lr=1e-2
        )
        
        # Training state
        self.training_history = []
        self.current_pivotal_states = set()
        
    def encode_trajectories(self, trajectories):
        batch_states = []
        batch_actions = []
        batch_discrete_actions = []  # ← Add this
        
        for trajectory in trajectories:
            states = [state for state, action in trajectory]
            actions = [action for state, action in trajectory]
            
            state_tensor = torch.tensor(states, dtype=torch.float32)
            action_tensor = torch.tensor(actions, dtype=torch.long)
            
            batch_states.append(state_tensor)
            batch_actions.append(action_tensor)
            batch_discrete_actions.append(action_tensor)  # ← Store discrete actions
        
        # Pad sequences
        max_len = max(len(states) for states in batch_states)
        
        padded_states = torch.zeros(len(trajectories), max_len, 2)
        padded_actions = torch.zeros(len(trajectories), max_len, dtype=torch.long)
        padded_discrete = torch.zeros(len(trajectories), max_len, dtype=torch.long)  # ← Add
        seq_lengths = []
        
        for i, (states, actions, discrete) in enumerate(zip(batch_states, batch_actions, batch_discrete_actions)):
            seq_len = len(states)
            padded_states[i, :seq_len] = states
            padded_actions[i, :seq_len] = actions
            padded_discrete[i, :seq_len] = discrete  # ← Pad discrete actions
            seq_lengths.append(seq_len)
        
        encoded_states = self.state_encoder(padded_states.to(self.device))
        encoded_actions = self.action_encoder(padded_actions.to(self.device))
        
        return encoded_states, encoded_actions, torch.tensor(seq_lengths), padded_discrete.to(self.device)
        #                                                                    ↑ Return discrete actions
    
    def compute_elbo_loss(self, states, actions, seq_lengths, original_discrete_actions, kl_weight ):
        """
        Compute the Evidence Lower Bound (ELBO) loss with all regularization terms.
        
        Args:
            states: Encoded states [batch_size, seq_len, state_dim]
            actions: Encoded actions [batch_size, seq_len, action_dim]  
            seq_lengths: Actual sequence lengths for each trajectory
            
        Returns:
            Dict containing all loss components
        """
        batch_size, seq_len = states.shape[:2]
        
        # 1. Prior network: get Beta parameters
        alpha_prior, beta_prior = self.prior_net(states)  # [batch_size, seq_len]
        
        # 2. Inference network: get HardKuma parameters
        alpha_posterior = self.inference_net(states, actions)  # [batch_size, seq_len]
        
        # 3. Sample binary latents using HardKumaraswamy
        hardkuma_dist = HardKumaraswamy(alpha_posterior)
        z_samples = hardkuma_dist.rsample()  # [batch_size, seq_len]
        
        # 4. Mask states based on binary samples (avoid inplace operations)
        z_expanded = z_samples.unsqueeze(-1).expand(-1, -1, self.state_dim)
        masked_states = torch.mul(states, z_expanded)  # [batch_size, seq_len, state_dim]
        
        # 5. Generation network: reconstruct actions
        action_logits = self.generation_net(masked_states)  # [batch_size, seq_len, action_vocab_size]
        
        # 6. Compute reconstruction loss
        action_logits_flat = action_logits.view(-1, self.action_vocab_size)
        original_actions_flat = original_discrete_actions.view(-1)  # ← Use parameter
        
        reconstruction_loss_flat = F.cross_entropy(
            action_logits_flat,
            original_actions_flat,
            reduction='none'
        )
        reconstruction_loss = reconstruction_loss_flat.view(batch_size, seq_len)
        
        # Apply sequence length masking
        mask = torch.arange(seq_len, device=self.device).unsqueeze(0) < seq_lengths.unsqueeze(1).to(self.device)
        masked_recon_loss = torch.where(mask, reconstruction_loss, torch.zeros_like(reconstruction_loss))
        reconstruction_loss = masked_recon_loss.sum() / mask.sum().clamp(min=1)
        
        # 7. Compute KL divergence
        beta_dist = BetaDistribution(alpha_prior, beta_prior)
        kl_divergence = hardkuma_dist.kl_divergence(beta_dist)
        masked_kl = torch.where(mask, kl_divergence, torch.zeros_like(kl_divergence))
        raw_kl = masked_kl.sum() / mask.sum().clamp(min=1)

        # After computing KL divergence

        #print(f"  Raw KL: {raw_kl:.6f}")  # DEBUG
        kl_divergence = torch.clamp(raw_kl, min=0.01)  # Then floor
        
 
        
        # 8. Compute L0 regularization (sparsity constraint)
        expected_l0 = hardkuma_dist.expected_l0_norm()
        l0_loss = torch.pow(expected_l0 - self.mu0, 2)
        
        # 9. Compute LT regularization (transition constraint) 
        # Encourage isolated activations - transitions between 0 and 1 should be ~2*mu0
        z_diff = torch.abs(z_samples[:, 1:] - z_samples[:, :-1])  # [batch_size, seq_len-1]
        seq_mask = torch.arange(seq_len-1, device=self.device).unsqueeze(0) < (seq_lengths-1).unsqueeze(1).to(self.device)
        
        masked_transitions = torch.where(seq_mask, z_diff, torch.zeros_like(z_diff))
        expected_transitions = masked_transitions.sum() / seq_mask.sum().clamp(min=1)
        lt_loss = torch.pow(expected_transitions - 2 * self.mu0, 2)
        
        # 10. Combine all losses using Lagrangian multipliers
        total_loss = (
            reconstruction_loss + 
            kl_weight * torch.abs(self.lambda_kl) * kl_divergence + 
            torch.abs(self.lambda_l0) * l0_loss + 
            torch.abs(self.lambda_lt) * lt_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_divergence': kl_divergence,
            'l0_loss': l0_loss,
            'lt_loss': lt_loss,
            'expected_l0': expected_l0,
            'expected_transitions': expected_transitions,
            'z_samples': z_samples.detach(),  # Detach for logging
            'alpha_prior': alpha_prior.detach(),
            'alpha_posterior': alpha_posterior.detach()
        }
    
    def train_step(self, trajectories,kl_weight ):
        
        # Forward pass
        states, actions, seq_lengths, discrete_actions = self.encode_trajectories(trajectories)
        loss_dict = self.compute_elbo_loss(states, actions, seq_lengths, discrete_actions, kl_weight)
        
        # Compute both gradients BEFORE any optimizer steps
        self.vae_optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        
        # VAE gradients (minimize loss)
        vae_params = (
            list(self.state_encoder.parameters()) + 
            list(self.action_encoder.parameters()) + # <-- MODIFIED: Included action_encoder parameters
            list(self.prior_net.parameters()) +
            list(self.inference_net.parameters()) +
            list(self.generation_net.parameters())
        )
        vae_grads = torch.autograd.grad(
            loss_dict['total_loss'], vae_params, 
            retain_graph=True, create_graph=False
        )
        
        # Lambda gradients 
        lambda_grads = torch.autograd.grad(
            loss_dict['total_loss'], 
            [self.lambda_kl, self.lambda_l0, self.lambda_lt],
            create_graph=False
        )
        
        # Apply VAE gradients manually
        for param, grad in zip(vae_params, vae_grads):
            param.grad = grad
        torch.nn.utils.clip_grad_norm_(vae_params, max_norm=1.0)
        self.vae_optimizer.step()
        
        # Apply lambda gradients manually
        for param, grad in zip([self.lambda_kl, self.lambda_l0, self.lambda_lt], lambda_grads):
            param.grad = grad
        self.lambda_optimizer.step()
        
        # Clamp lambdas
        with torch.no_grad():
            self.lambda_kl.data.clamp_(0.1, 10.0)
            self.lambda_l0.data.clamp_(0.01, 5.0)
            self.lambda_lt.data.clamp_(0.01, 5.0)
        
        # Logging
        logged_losses = {
            k: v.detach().item() if torch.is_tensor(v) and v.numel() == 1 else v
            for k, v in loss_dict.items()
            if k not in ['z_samples', 'alpha_prior', 'alpha_posterior']
        }
        
        return logged_losses
        
    def extract_pivotal_states(self, trajectories: List[List[Tuple]], threshold_percentile: float = 80) -> List[Tuple]:
        """
        Extract pivotal states after training by analyzing prior means.
        
        Args:
            trajectories: All trajectories to analyze
            threshold_percentile: Percentile threshold for pivotal state selection (paper uses top 20%)
            
        Returns:
            List of pivotal state coordinates
        """
        self.eval()
        
        state_importance_scores = {}
        
        with torch.no_grad():
            for trajectory in trajectories:
                states, _, seq_lengths, _ = self.encode_trajectories([trajectory])
                
                # Get prior importance scores
                alpha_prior, beta_prior = self.prior_net(states)
                
                # Prior mean = alpha / (alpha + beta) 
                prior_means = alpha_prior / (alpha_prior + beta_prior)
                
                # Extract original coordinates and their importance scores
                original_states = [state for state, action in trajectory]
                seq_len = len(original_states)
                
                for i, (x, y) in enumerate(original_states[:seq_len]):
                    coord = (x, y)
                    importance = prior_means[0, i].item()
                    
                    if coord not in state_importance_scores:
                        state_importance_scores[coord] = []
                    state_importance_scores[coord].append(importance)
        
        # Average importance scores across all occurrences
        avg_importance = {
            coord: np.mean(scores) 
            for coord, scores in state_importance_scores.items()
        }
        
        # Select top threshold_percentile% as pivotal states
        threshold_percentage=100-threshold_percentile
        sorted_states = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        num_pivotal = max(1, int(len(sorted_states) * (threshold_percentage / 100)))
        
        pivotal_states = [coord for coord, score in sorted_states[:num_pivotal]]
        
        self.current_pivotal_states = set(pivotal_states)
        return pivotal_states
    
    def train(
        self, 
        stat_buffer, 
        num_epochs: int = 100,
        batch_size: int = 8,
        convergence_threshold: float = 1e-4,
        patience: int = 10,
        initial_kl_weight: float = 0.0
    ) -> List[Tuple]:
        """
        Full training loop for pivotal state discovery.
        
        Args:
            stat_buffer: StatBuffer containing trajectory data
            num_epochs: Maximum number of training epochs
            batch_size: Number of trajectories per batch
            convergence_threshold: Threshold for early stopping
            patience: Number of epochs to wait for improvement
            
        Returns:
            List of discovered pivotal states
        """
        print(f"Starting VAE training on {stat_buffer.episodes_in_buffer} episodes...")
        
        # Get all trajectories from stat buffer
        all_trajectories = stat_buffer.get_STATE_ACT_traj_stream_byep()
        
        best_loss = float('inf')
        patience_counter = 0

         # <-- MODIFIED: Initialize KL annealing schedule
        kl_weight = initial_kl_weight
        self.current_kl_weight = kl_weight  # Store for curiosity
        # Anneal over the first 50% of epochs
        anneal_epochs = max(1, num_epochs * 0.75)
        annealing_rate = 1.0 / anneal_epochs
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Sample batches of trajectories
            num_batches = max(1, len(all_trajectories) // batch_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_trajectories))
                batch_trajectories = all_trajectories[start_idx:end_idx]
                
                # Train step
                loss_dict = self.train_step(batch_trajectories,kl_weight)
                epoch_losses.append(loss_dict)
            
            # Average losses for the epoch
            avg_losses = {}
            for key in epoch_losses[0].keys():
                avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
            
            # Store training history
            self.training_history.append(avg_losses)
            
            # Update KL weight
            kl_weight = min(1.0, kl_weight + annealing_rate)
            self.current_kl_weight = kl_weight  # ← ADD THIS


            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={avg_losses['total_loss']:.4f}, "
                      f"Recon={avg_losses['reconstruction_loss']:.4f}, "
                      f"KL={avg_losses['kl_divergence']:.4f}, "
                      f"L0={avg_losses['expected_l0']:.2f}, "
                      f"KL_Weight={kl_weight:.2f}") # <-- MODIFIED: Log the weight
            
            # Check for convergence
            current_loss = avg_losses['total_loss']
            if current_loss < best_loss - convergence_threshold:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Converged after {epoch} epochs (patience={patience})")
                break
        
        # Extract final pivotal states
        pivotal_states = self.extract_pivotal_states(all_trajectories)
        
        print(f"Training completed. Discovered {len(pivotal_states)} pivotal states:")
        for i, (x, y) in enumerate(pivotal_states[:10]):  # Show first 10
            print(f"  {i+1}: ({x}, {y})")
        if len(pivotal_states) > 10:
            print(f"  ... and {len(pivotal_states) - 10} more")
        
        return pivotal_states
    
    def eval(self):
        """Set all networks to evaluation mode."""
        self.state_encoder.eval()
        self.action_encoder.eval()
        self.prior_net.eval()
        self.inference_net.eval()
        self.generation_net.eval()
    
    def train_mode(self):
        """Set all networks to training mode."""
        self.state_encoder.train()
        self.action_encoder.train()
        self.prior_net.train()
        self.inference_net.train()
        self.generation_net.train()


    def compute_curiosity_reward(self, episode_states: List[Tuple[int, int]], scale_factor: float = 0.01) -> float:
        """
        Compute curiosity reward based on VAE reconstruction error.
        High reconstruction error = novel area = high curiosity reward.
        
        Args:
            episode_states: List of (x, y) coordinates visited in current episode
            scale_factor: Scaling factor for curiosity rewards (reduced from 0.05 to 0.01)
            
        Returns:
            float: Curiosity reward value
        """
        if len(episode_states) == 0:
            return 0.0
        
        # Filter out any invalid positions that might cause crashes
        valid_states = []
        for x, y in episode_states:
            # Ensure positions are within valid maze bounds (1 to size-2)
            if 1 <= x <= 8 and 1 <= y <= 8:  # For 10x10 maze, valid range is 1-8
                valid_states.append((x, y))
        
        if len(valid_states) == 0:
            return 0.0
        
        # Create fake trajectory for VAE processing
        dummy_actions = [0] * len(valid_states)  # Use action 0 (turn_left) as dummy
        fake_trajectory = list(zip(valid_states, dummy_actions))
        
        try:
            # Set to evaluation mode to avoid training
            self.eval()
            
            with torch.no_grad():
                # Encode the trajectory
                states, actions, seq_lengths, discrete = self.encode_trajectories([fake_trajectory])
                
                # Compute reconstruction loss (without gradients)
                loss_dict = self.compute_elbo_loss(states, actions, seq_lengths, discrete, self.current_kl_weight)
                reconstruction_error = loss_dict['reconstruction_loss'].item()
                
                # Reduced scale and cap for better balance
                scale_factor = 0.1
                curiosity_reward = min(2.0, reconstruction_error * scale_factor)  # Cap at 0.05 instead of 0.2
                
                return curiosity_reward
                
        except Exception as e:
            # If VAE computation fails, return zero curiosity
            print(f"    Curiosity computation failed: {e}")
            return 0.0
        
        finally:
            # Return to training mode
            self.train_mode()