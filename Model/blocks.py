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



class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_channels, num_heads,
                     t_emb_dim, is_down_sample, num_layers, is_attn, 
                     cross_attn = False, context_dim = None, context_dim_image=None):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.num_layers = num_layers
        self.is_down_sample = is_down_sample
        self.is_attn = is_attn
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        
        # Resenet Block
        self.resnet_block_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(num_groups=norm_channels, num_channels = in_channels if i ==0  else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size = 3, stride = 1, padding = 1)
                    )
                for i in range(num_layers)
            ]
        )

        # Time Embedding Block
        if self.t_emb_dim is not None:
            self.t_emb_block = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.SiLU(),
                        nn.Linear(self.t_emb_dim, out_channels)
                    )
                    for i in range(num_layers)
                ]
            )

        # Resenet Second Block
        self.resnet_block_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels,
                               kernel_size = 3, stride = 1, padding = 1)
                )
                for _ in range(num_layers)
            ]
        )

        # Attention Block
        if self.is_attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first = True )
                    for _ in range(num_layers)
                ]
            )
        

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            # This can be used for all the Three modules Text, Image and Style 
            # since we are changing the context_dim of each to out channels
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            # Context Projection for Text, Image and Style in that order below
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
            self.context_proj_image = nn.ModuleList(
                [nn.Linear(context_dim_image, out_channels)
                 for _ in range(num_layers)]
            )
            # self.context_proj_style = nn.ModuleList(
            #     [nn.Linear(context_dim, out_channels)
            #      for _ in range(num_layers)]
            # )

        # Downsample Block
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i ==0 else out_channels, out_channels, stride = 1)
                for i in range(num_layers)
            ]
        )

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.is_down_sample else nn.Identity()
    def forward(self, x, t_emb = None, context = None,context_image = None, context_style = None):
        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_block_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_block[i](t_emb)[:, :, None, None]
            out = self.resnet_block_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            if self.attn:
                # Attention Block
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                # we are doing the transpose because
                # This is necessary because the attention mechanism operates over the sequence dimension, and the nn.MultiheadAttention expects the sequence length to be the second dimension
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
            
            # TODO Cross Attention [image and Style need to add]
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                context_proj_image = self.context_proj_image[i](context_image)
                # context_proj_style = self.context_proj_style[i](context_style)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn_image, _ = self.cross_attentions[i](in_attn, context_proj_image, context_proj_image)
                # out_attn_style, _ = self.cross_attentions[i](in_attn, context_proj_style, context_proj_style)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out_attn_image = out_attn_image.transpose(1, 2).reshape(batch_size, channels, h, w)
                # out_attn_style = out_attn_style.transpose(1, 2).reshape(batch_size, channels, h, w)
                # out = out + out_attn + out_attn_image + out_attn_style
                out = out + out_attn + out_attn_image 

        # Downsample
        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    r"""
    Mid conv block with attention.
    Sequence of following blocks
    1. Resnet block with time embedding
    2. Attention block
    3. Resnet block with time embedding
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim, 
                 num_heads, num_layers, norm_channels, 
                 cross_attn= False, context_dim = None, context_dim_image = None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers + 1)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers + 1)
            ])
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
             for _ in range(num_layers)]
        )

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
            self.context_proj_image = nn.ModuleList(
                [nn.Linear(context_dim_image, out_channels)
                 for _ in range(num_layers)]
            )
            # self.context_proj_style = nn.ModuleList(
            #     [nn.Linear(context_dim, out_channels)
            #      for _ in range(num_layers)]
            # )
    
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
             for _ in range(num_layers)]
        )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers + 1)
            ]
        )
    
    def forward(self, x, t_emb=None, 
                context=None, context_image = None, context_style = None):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                context_proj_image = self.context_proj_image[i](context_image)
                # context_proj_style = self.context_proj_style[i](context_style)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn_image, _ = self.cross_attentions[i](in_attn, context_proj_image, context_proj_image)
                # out_attn_style , _ = self.cross_attentions[i](in_attn, context_proj_style, context_proj_style)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out_attn_image = out_attn_image.transpose(1, 2).reshape(batch_size, channels, h, w)
                # out_attn_style = out_attn_style.transpose(1, 2).reshape(batch_size, channels, h, w)
                # out = out + out_attn + out_attn_image + out_attn_style
                out = out + out_attn + out_attn_image 
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)
        
        return out

# TODO : Complete the CrossAttention part of the UpBlock

class UpBlock(nn.Module):
    r"""
    Up conv block with attention.
    Sequence of following blocks
    1. Upsample
    1. Concatenate Down block output
    2. Resnet block with time embedding
    3. Attention Block
    """
    
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 up_sample, num_heads, num_layers, attn, norm_channels,
                 cross_attn=False, context_dim = None, context_dim_image = None):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.cross_attn = cross_attn
        self.context_dim = context_dim
        self.attn = attn
        self.resnet_conv_first = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
                for _ in range(num_layers)
            ])
        
        self.resnet_conv_second = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
        
        # TODO Cross Attention [image and Style need to add]
        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )
            self.context_proj_image = nn.ModuleList(
                [nn.Linear(context_dim_image, out_channels)
                 for _ in range(num_layers)]
            )
            # self.context_proj_style = nn.ModuleList(
            #     [nn.Linear(context_dim, out_channels)
            #      for _ in range(num_layers)]
            # )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels,
                                                 4, 2, 1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x, out_down=None, t_emb=None,
                context = None, context_image = None, context_style = None):
        # Upsample
        x = self.up_sample_conv(x)
        
        # Concat with Downblock output
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            # Self Attention
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

            # TODO Cross Attention [image and Style need to add]
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used"
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert len(context.shape) == 3, \
                    "Context shape does not match B,_,CONTEXT_DIM"
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim,\
                    "Context shape does not match B,_,CONTEXT_DIM"
                context_proj = self.context_proj[i](context)
                context_proj_image = self.context_proj_image[i](context_image)
                # context_proj_style = self.context_proj_style[i](context_style)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn_image, _ = self.cross_attentions[i](in_attn, context_proj_image, context_proj_image)
                # out_att_style, _ = self.cross_attentions[i](in_attn, context_proj_style, context_proj_style)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out_attn_image = out_attn_image.transpose(1, 2).reshape(batch_size, channels, h, w)
                # out_att_style = out_att_style.transpose(1, 2).reshape(batch_size, channels, h, w)
                # out = out + out_attn + out_attn_image + out_att_style
                out = out + out_attn + out_attn_image 
        return out