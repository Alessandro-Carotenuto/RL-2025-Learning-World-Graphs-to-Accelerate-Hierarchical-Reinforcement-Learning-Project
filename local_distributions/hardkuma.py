import torch


#----------------------------------------------------------------------------#
#                      HARD KUMARASWAMY DISTRIBUTION                        #
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
            
        u = torch.rand(sample_shape, device=self.alpha.device)
        
    # INVERSE CDF FOR KUMARASWAMY (BETA=1)
        eps = 1e-8  # EPSILON FOR NUMERICAL STABILITY
        x = torch.pow(1 - u + eps, 1.0 / (self.alpha + eps))
        
    # STRETCH OPERATION
        s = x * (self.zeta - self.gamma) + self.gamma
        
    # RECTIFY TO [0, 1] INTERVAL
        z = torch.clamp(s, 0, 1)
        
        return z
        
    def kl_divergence(self, other_dist):
        # KL DIVERGENCE: Bernoulli surrogate for HardKuma vs Beta prior.
        # The continuous Kumaraswamy-Beta KL approximation produces systematically
        # negative values (clipped to 0 by relu), giving zero gradient signal.
        # Instead we compute KL(Bernoulli(q) || Bernoulli(p)) where:
        #   q = P(z=1) from HardKuma expected value
        #   p = alpha/(alpha+beta) = mean of the Beta prior
        # This is always >= 0 and provides correct gradient to the inference network.
        eps = 1e-8
        q = self._expected_value().clamp(eps, 1.0 - eps)
        p = (other_dist.alpha / (other_dist.alpha + other_dist.beta)).clamp(eps, 1.0 - eps)
        return q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
        
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
    

#----------------------------------------------------------------------------#
#                          BETA DISTRIBUTION                                 #
#----------------------------------------------------------------------------#
class BetaDistribution:
    # SIMPLE BETA DISTRIBUTION FOR KL WITH HARDKUMA
    
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta