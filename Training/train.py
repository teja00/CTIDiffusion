import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from wsgiref import validate
import torch
from PIL import Image
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from Utils.iam_dataset import IAMDataset
from Utils.config_utils import get_config_value, validate_image_config, validate_text_config
from Utils.diffusion_utils import drop_image_condition, drop_text_condition
from Utils.pre_trained_utils import get_text_representation, get_tokenizer_and_model, get_image_model_processor, get_image_representation
from Model.Unet import Unet
from Model.NoiseScheduler import NoiseScheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# TODO Style component for training we need to do

def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = NoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    text_tokenizer = None
    text_model = None
    empty_text_embed = None
    # emtpy_style_embed = None
    empty_image_embed = None
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, "condition_types missing in condition_config"
        condition_types = condition_config['condition_types']
        if ('text') in condition_types:
            validate_text_config(condition_config)
            # TODO: we need to create a text model as below
            text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']['text_embed_model'], device=device)
            empty_text_embed = get_text_representation([''], text_tokenizer, text_model, device)
        if 'image' in condition_types:
            validate_image_config(condition_config)
            empty_image_embed = torch.zeros((3, dataset_config['im_size'], dataset_config['im_size']), device=device)


    # TODO: we need to create a dataset Class as below
    im_dataset_cls = {
        'IAMHandwriting': IAMDataset,
    }.get(dataset_config['name'])
    
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                condition_config=condition_config)
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True)
    
    # Instantiate the model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.train()
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    

    for epoch_idx in range(num_epochs):
        losses = []
        for data in tqdm(data_loader):
            optimizer.zero_grad()
            cond_input = None
            if condition_config is not None:
                im, cond_input = data
            else:
                im = data
            im = im.float().to(device)

            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, "Text condition missing in cond_input"
                    validate_text_config(condition_config=condition_config)
                    text_condition = get_text_representation(cond_input['text'], text_tokenizer, text_model, device)
                    text_drop_prob = get_config_value(condition_config['text_condition_config'], 'cond_drop_prob', 0.0)
                    text_condition = drop_text_condition(text_condition, im, empty_text_embed, text_drop_prob)
                    cond_input['text'] = text_condition
            if 'image' in condition_types:
                with torch.no_grad():
                    assert 'image' in cond_input, "Image condition missing in cond_input"
                    validate_image_config(condition_config=condition_config)
                    image_condition = cond_input['image'].to(device)  # It's already a tensor, just move to device
                    # image_drop_prob = get_config_value(condition_config['image_condition_config'], 'cond_drop_prob', 0.0)
                    # image_condition = drop_image_condition(image_condition, empty_image_embed, image_drop_prob)
                    cond_input['image'] = image_condition
                        
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name']))
    
    print('Done Training ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/IAM.yaml', type=str)
    args = parser.parse_args()
    train(args)