from cgitb import text
import pickle
import glob
import os
import torch


def drop_text_condition(text_embed, im, empty_text_embed,text_drop_prob):
    if text_drop_prob > 0:
        text_drop_mask = torch.zeros((im.shape[0]), device=im.device).float().uniform_(0, 1) < text_drop_prob
        assert empty_text_embed is not None, "Empty text embed is not provided"
        text_embed[text_drop_mask, :, :] = empty_text_embed[0]
    return text_embed



def drop_image_condition(image_embed, empty_image_embed, image_drop_prob):
    if image_drop_prob > 0:
        # Create a drop mask based on the drop probability
        image_drop_mask = torch.zeros((image_embed.shape[0]), device=image_embed.device).float().uniform_(0, 1) < image_drop_prob
        
        # Ensure empty_image_embed is provided
        assert empty_image_embed is not None, "Empty image embed is not provided"
        
        # Apply the mask to replace selected embeddings with the empty embedding
        image_embed[image_drop_mask, :] = empty_image_embed[0]
    
    return image_embed