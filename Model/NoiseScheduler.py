import torch

class NoiseScheduler:
    r"""
    This Class is Created to Implement the Noise Scheduler 
    for the Diffusion Model we are trying to Implement.
    """
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = (torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)
    
    def add_noise(self, original_image, noise, t):
        r"""
        In the Diffusion Process we are trying to Implement, 
        This method is used when we are adding noise to the Image.
        :param original_image: The Original Image to which we are adding noise.
        :param noise: The Noise to be added to the Image. [Gaussian Noise]
        :param t: timestep of the forward process which is having a shape of -> (B, )
        """
        original_shape = original_image.shape
        batch_size = original_shape[0]
        
        device = original_image.device
        t = t.to(device)
        
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].to(device).reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].to(device).reshape(batch_size)

        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)
        
        return (sqrt_alpha_cum_prod * original_image 
                + sqrt_one_minus_alpha_cum_prod * noise)

    def sample_from_prev_timestep(self, x_t, noise_pred, t):
        r"""
        We will use the Noise Prediction predicted from the Diffusion Model 
        to sample the x_{t - 1} from the x_t.
        :param x_t: The Image at time t.
        :param noise_pred: The Noise Prediction at time t.
        """
        device = x_t.device
        t = t.to(device)

        x_0 = ((x_t - (self.sqrt_one_minus_alpha_cum_prod[t] * noise_pred)) /
              self.sqrt_alpha_cum_prod[t])

        x_0 = torch.clamp(x_0, -1., 1.)

        mean = x_t - ((self.betas[t] * noise_pred) / self.sqrt_one_minus_alpha_cum_prod[t])
        mean = mean / torch.sqrt(self.alphas[t])

        if t == 0:
            return mean, x_0
        else:
            variance = (1 - self.alpha_cum_prod[t - 1]) / (1.0 - self.alpha_cum_prod[t])
            variance = variance * self.betas[t]
            sigma = variance ** 0.5
            z = torch.randn(x_t.shape).to(device)
            
            return mean + sigma * z, x_0