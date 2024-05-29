import torch
import torch.nn as nn

def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the 
    sinusoidal time embedding formula.
    :param time_steps: The time steps tensor.
    :param temb_dim: The dimension of the time embedding.
    :return: B X D embedding representation of the time steps.
    """
    assert temb_dim % 2 == 0, "The dimension of the time embedding should be even."

    factor = 10000 ** ((torch.arange(
        start = 0 , end = temb_dim // 2, dtype = torch.float32, device = time_steps.device) / (temb_dim // 2) )
        )

    t_emb = time_steps[:,None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim = -1)
    return t_emb