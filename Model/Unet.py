import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from multiprocessing import context
import torch
import torch.nn as nn
from Model.blocks import get_time_embedding
from Model.blocks import DownBlock, MidBlock, UpBlock
from Utils.config_utils import get_config_value

# This is just the DDPM Implementation of the UNET
# Need to remember that in forward we are only sending one Context in which all the image, text and style are included

class Unet(nn.Module):
    r"""
    Unet model comprising
    Down blocks, Midblocks and Uplocks
    """
    
    def __init__(self, im_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        self.attns = model_config['attn_down']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']
        self.conv_out_channels = model_config['conv_out_channels']
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1

        ## Image, Text, Conditioning config

        self.text_cond = False
        self.image_cond = False
        # self.style_cond = False
        self.text_embed_dim = None
        self.image_embed_dim = None
        self.condition_config = get_config_value(model_config, 'condition_config', None)
        if self.condition_config is not None:
            assert 'condition_types' in self.condition_config, "condition_types missing in condition_config"
            condition_types = self.condition_config['condition_types']
            if 'text' in condition_types:
                self.text_cond = True
                # self.style_cond = True
                self.text_embed_dim = self.condition_config['text_condition_config']['text_embed_dim']
            if 'image' in condition_types:
                self.image_cond = True
                self.image_embed_dim = self.condition_config['image_condition_config']['image_embed_dim']

        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        self.cond = self.text_cond or self.image_cond  
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        is_down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        is_attn=self.attns[i], norm_channels=self.norm_channels,
                                        context_dim = self.text_embed_dim,
                                        context_dim_image=self.image_embed_dim))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels,
                                      context_dim = self.text_embed_dim,
                                      context_dim_image=self.image_embed_dim))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                                    self.t_emb_dim, up_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_up_layers,
                                        norm_channels=self.norm_channels,
                                        context_dim=self.text_embed_dim,
                                        context_dim_image=self.image_embed_dim))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, im_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, cond_input= None):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        if self.text_cond:
            assert 'text' in cond_input, "Text condition missing"
            context_hidden_states = cond_input['text']
        if self.image_cond:
            assert 'image' in cond_input, "Image condition missing"
            context_hidden_states_image = cond_input['image']
        # if self.style_cond:
        #     assert 'style' in cond_input, "Style condition missing"
        #     context_hidden_states_style = cond_input['style']
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            # out = down(out, t_emb, context_hidden_states, context_hidden_states_image, context_hidden_states_style)
            out = down(out, t_emb, context_hidden_states, context_hidden_states_image)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for mid in self.mids:
            # out = mid(out, t_emb, context_hidden_states, context_hidden_states_image, context_hidden_states_style)
            out = mid(out, t_emb, context_hidden_states, context_hidden_states_image)

        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            # out = up(out, down_out, t_emb, context_hidden_states, context_hidden_states_image, context_hidden_states_style)
            out = up(out, down_out, t_emb, context_hidden_states, context_hidden_states_image)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out
