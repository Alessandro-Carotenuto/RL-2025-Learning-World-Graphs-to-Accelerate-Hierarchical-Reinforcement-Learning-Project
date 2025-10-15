import torch
from torch.distributions import Uniform
from torch import digamma, lgamma
import torch.nn.functional as F

import numpy as np


# HARD KUMARASWAMY DISTRIBUTION:
# DIFFERENTIABLE BINARY SAMPLING WITH STRETCH-AND-RECTIFY FOR VAE STATE SELECTION
# ENABLES END-TO-END TRAINING WHILE MAINTAINING DISCRETE DECISIONS

#----------------------------------------------------------------------------#
#                            Hard Kumaraswamy                                #
#----------------------------------------------------------------------------#


class HardKumaraswamy:
    def __init__(self, alpha, beta=1.0, gamma=-0.1, zeta=1.1):
        # INITIALIZE HARD KUMARASWAMY DISTRIBUTION PARAMETERS
        self.alpha = alpha
        self.beta = beta  # Fixed at 1 in the paper
        self.gamma = gamma  # Stretch parameters from Bastings et al.
        self.zeta = zeta
        
    def rsample(self, sample_shape=None):
        # SAMPLE FROM HARD KUMARASWAMY USING STRETCH-RECTIFY
        if sample_shape is None:
            sample_shape = self.alpha.shape
            
    # SAMPLE UNIFORM NOISE
        uniform = Uniform(0, 1)
        u = uniform.sample(sample_shape).to(self.alpha.device)
        
    # INVERSE CDF FOR KUMARASWAMY (BETA=1)
        eps = 1e-8  # EPSILON FOR NUMERICAL STABILITY
        x = torch.pow(1 - u + eps, 1.0 / (self.alpha + eps))
        
    # STRETCH OPERATION
        s = x * (self.zeta - self.gamma) + self.gamma
        
    # RECTIFY TO [0, 1] INTERVAL
        z = torch.clamp(s, 0, 1)
        
        return z
        
    def log_prob(self, value):
        # COMPUTE LOG PROBABILITY OF OBSERVED VALUE
    # PROBABILITY OF GETTING 1
        threshold = (1.0 - self.gamma) / (self.zeta - self.gamma)
        
    # KUMARASWAMY CDF
        prob_one = torch.pow(torch.clamp(1.0 - threshold, 0.0, 1.0), self.alpha)
        prob_zero = 1.0 - prob_one
        
    # LOG PROBABILITY BASED ON VALUE
        eps = 1e-8
        log_prob_one = torch.log(prob_one + eps)
        log_prob_zero = torch.log(prob_zero + eps)
        
        log_prob = torch.where(value > 0.5, log_prob_one, log_prob_zero)
        return log_prob
        
    def kl_divergence(self, other_dist):
        # KL DIVERGENCE BETWEEN KUMARASWAMY AND BETA DISTRIBUTIONS
    # BETA DISTRIBUTION PARAMETERS
        alpha_beta = other_dist.alpha
        beta_beta = other_dist.beta
        
    # KUMARASWAMY DISTRIBUTION PARAMETER
        alpha_kuma = self.alpha
        
    # DIGAMMA TERMS FOR NORMALIZATION
        psi_alpha = digamma(alpha_beta)
        psi_beta = digamma(beta_beta)
        psi_alpha_plus_beta = digamma(alpha_beta + beta_beta)
        
    # LOG-GAMMA TERMS FOR NORMALIZATION
        log_beta_normalizer = lgamma(alpha_beta) + lgamma(beta_beta) - lgamma(alpha_beta + beta_beta)

    # COMBINE ALL TERMS FOR KL
        eps = 1e-8
        
        term1 = (1 - 1 / (alpha_kuma + eps)) * (psi_alpha - psi_alpha_plus_beta)
        term2 = torch.log(alpha_kuma + eps) - psi_beta + psi_alpha_plus_beta
        term3 = (beta_beta - 1) * torch.log(alpha_kuma + eps) # Simplified term for Kuma beta=1
        
    # SIMPLIFIED KL APPROXIMATION
        kl = (
            (alpha_beta - 1) * (-1 / (alpha_kuma + eps)) +
            (beta_beta - 1) * (torch.log(alpha_kuma + eps) - np.euler_gamma) -
            log_beta_normalizer -
            torch.log(alpha_kuma + eps)
        )
        
    # KL SHOULD BE NON-NEGATIVE
        return F.relu(kl)
        
    def expected_l0_norm(self):
        # EXPECTED L0 NORM FOR SPARSITY
    # EXPECTED VALUE OF BINARY OUTPUT
        expected_value = self._expected_value()
        return expected_value.sum()
        
    def _expected_value(self):
        # EXPECTED VALUE OF HARD KUMARASWAMY
    # KUMARASWAMY EXPECTED VALUE
        kuma_expected = 1.0 / (1.0 + self.alpha)
        
    # APPLY STRETCH TRANSFORMATION
        stretched_expected = kuma_expected * (self.zeta - self.gamma) + self.gamma
        
    # RECTIFY TO [0, 1] INTERVAL
        rectified_expected = torch.clamp(stretched_expected, 0, 1)
        
        return rectified_expected
    
#------------------------------------------------------------------------------   
    

# Beta distribution class for KL divergence computation
class BetaDistribution:
    # SIMPLE BETA DISTRIBUTION FOR KL WITH HARDKUMA
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta