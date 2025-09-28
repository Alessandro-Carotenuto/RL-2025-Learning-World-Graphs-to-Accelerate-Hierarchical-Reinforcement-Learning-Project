import torch
from torch.distributions import Uniform


# Hard Kumaraswamy Summary:
# VAE needs binary decisions (keep/skip each state) but requires differentiable 
# training. Hard Kumaraswamy distribution samples exact 0s/1s while maintaining 
# gradients through stretch-rectify procedure. Two networks trained 
# simultaneously: prior pψ(zt|st) = Beta(αt, βt) learns state importance, 
# posterior qφ(zt|at,st) = HardKuma(α̃t, 1) uses action evidence for 
# reconstruction. ELBO objective balances reconstruction quality with KL 
# divergence between posterior and prior. After training, states with highest 
# prior means (top 20%) become pivotal states. This discovers which waypoints 
# are consistently important across many trajectories.

#----------------------------------------------------------------------------#
#                            Hard Kumaraswamy                                #
#----------------------------------------------------------------------------#


class HardKumaraswamy:
    def __init__(self, alpha, beta=1.0, gamma=-0.1, zeta=1.1):
        """
        Hard Kumaraswamy distribution for differentiable binary sampling.
        
        Args:
            alpha (torch.Tensor): Concentration parameter (learned)
            beta (float): Second parameter (fixed at 1 in paper)
            gamma (float): Lower stretch bound (typically -0.1)
            zeta (float): Upper stretch bound (typically 1.1)
        """
        self.alpha = alpha
        self.beta = beta  # Fixed at 1 in the paper
        self.gamma = gamma  # Stretch parameters from Bastings et al.
        self.zeta = zeta
        
    def rsample(self, sample_shape=None):
        """
        Reparameterized sampling using the stretch-and-rectify procedure.
        
        Args:
            sample_shape (tuple): Shape of samples to generate
            
        Returns:
            torch.Tensor: Binary samples (0 or 1)
        """
        if sample_shape is None:
            sample_shape = self.alpha.shape
            
        # Step 1: Sample uniform noise
        uniform = Uniform(0, 1)
        u = uniform.sample(sample_shape).to(self.alpha.device)
        
        # Step 2: Kumaraswamy inverse CDF (beta=1 case)
        # x = (1 - u)^(1/alpha)
        eps = 1e-8  # Small epsilon to avoid numerical issues
        x = torch.pow(1 - u + eps, 1.0 / (self.alpha + eps))
        
        # Step 3: Stretch operation
        # s = x * (zeta - gamma) + gamma
        s = x * (self.zeta - self.gamma) + self.gamma
        
        # Step 4: Rectify operation (hard clamp to [0,1])
        # z = max(0, min(1, s))
        z = torch.clamp(s, 0, 1)
        
        return z
        
    def log_prob(self, value):
        """
        Log probability computation for Hard Kumaraswamy.
        
        Args:
            value (torch.Tensor): Observed values (should be 0 or 1)
            
        Returns:
            torch.Tensor: Log probabilities
        """
        # Probability of getting 1 (P(s > 1))
        # P(s > 1) = P(x * (zeta - gamma) + gamma > 1)
        # P(x > (1 - gamma) / (zeta - gamma))
        threshold = (1.0 - self.gamma) / (self.zeta - self.gamma)
        
        # For Kumaraswamy(alpha, 1): P(X > t) = (1 - t)^alpha
        prob_one = torch.pow(torch.clamp(1.0 - threshold, 0.0, 1.0), self.alpha)
        prob_zero = 1.0 - prob_one
        
        # Log probability based on observed value
        eps = 1e-8
        log_prob_one = torch.log(prob_one + eps)
        log_prob_zero = torch.log(prob_zero + eps)
        
        log_prob = torch.where(value > 0.5, log_prob_one, log_prob_zero)
        return log_prob
        
    def kl_divergence(self, other_dist):
        """
        KL divergence with Beta distribution (closed form from Nalisnick & Smyth).
        
        Args:
            other_dist: Object with .alpha and .beta attributes (Beta distribution)
            
        Returns:
            torch.Tensor: KL divergence values
        """
        # This is a simplified approximation - proper implementation needs 
        # full closed-form from the referenced papers
        
        # Expected value of Hard Kumaraswamy
        hardkuma_mean = self._expected_value()
        
        # Expected value of Beta prior
        beta_mean = other_dist.alpha / (other_dist.alpha + other_dist.beta)
        
        # Simplified KL approximation using squared difference
        # Replace with proper closed-form KL for production use
        kl = torch.pow(hardkuma_mean - beta_mean, 2)
        
        return kl
        
    def expected_l0_norm(self):
        """
        Expected L0 norm for sparsity regularization.
        
        Returns:
            torch.Tensor: Expected number of non-zero elements
        """
        # E[z] where z is the binary output
        expected_value = self._expected_value()
        return expected_value.sum()
        
    def _expected_value(self):
        """
        Compute expected value of Hard Kumaraswamy distribution.
        
        Returns:
            torch.Tensor: Expected values
        """
        # For Kumaraswamy(alpha, 1): E[x] = 1/(1 + alpha)
        kuma_expected = 1.0 / (1.0 + self.alpha)
        
        # Apply stretch transformation
        stretched_expected = kuma_expected * (self.zeta - self.gamma) + self.gamma
        
        # Expected value after rectification (simplified approximation)
        # Proper implementation would require numerical integration
        rectified_expected = torch.clamp(stretched_expected, 0, 1)
        
        return rectified_expected
    
#------------------------------------------------------------------------------   
    

# Beta distribution class for KL divergence computation
class BetaDistribution:
    """Simple Beta distribution class for KL divergence with HardKuma."""
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta