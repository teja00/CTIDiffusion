from cgitb import text
import random
import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
from Model.Unet import Unet
from Model.NoiseScheduler import NoiseScheduler
from Utils.config_utils import get_config_value, validate_image_config, validate_text_config
from Utils.iam_dataset import IAMDataset
from Utils.pre_trained_utils import get_image_model_processor, get_text_representation, get_tokenizer_and_model, get_image_representation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## TODO Need to convert the sample to from DDPM to Conditional DDPM by adding the condition embeddings

def sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, text_tokenizer, text_model, image_model, image_processor):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2**sum(autoencoder_model_config['down_sample'])
    xt = torch.randn((train_config['num_samples'],
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)

    save_count = 0
    text_prompt = ['Teja']
    empty_prmopt = ['']
    text_prompt_embed = get_text_representation(text_prompt, text_tokenizer, text_model, device)
    empty_text_embed = get_text_representation(empty_prmopt, text_tokenizer, text_model, device)

    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    
    dataset = IAMDataset(split='train', 
                         im_path=dataset_config['im_path'],
                         im_size=dataset_config['im_size'],
                         im_channels=dataset_config['im_channels'],
                         condition_config=condition_config)
    
    index = random.randint(0, len(dataset))
    image = dataset.get(index).unsqueeze(0).to(device)
    image_embed = get_image_representation(image, image_model, image_processor, device)
    empty_image = Image.new('RGB', (dataset_config['im_size'], dataset_config['im_size']), color = (0, 0, 0))
    empty_image_embed = get_image_representation(empty_image,image_model, image_processor, device)
    uncond_input = {
        'text' : empty_text_embed,
        'image' : empty_image_embed
    }
    cond_input = {
        'text' : text_prompt_embed,
        'image' : image_embed
    }

    cf_guidance_scale = get_config_value(diffusion_model_config, key='cf_guidance_scale', default_value=0.0)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0], ))* i).long().to(device)

        noise_pred_cond = model(xt, t , cond_input)

        if(cf_guidance_scale > 1):
            noise_pred_uncond = model(xt, t, uncond_input)
            noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        ims = xt
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)
        
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
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
    
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    assert condition_config is not None, ("This sampling script is for image and text conditional "
                                          "but no conditioning config found")
    condition_types = get_config_value(condition_config, 'condition_types', [])
    assert 'text' in condition_types, ("This sampling script is for image and text conditional "
                                       "but no text condition found in config")
    assert 'image' in condition_types, ("This sampling script is for image and text conditional "
                                       "but no image condition found in config")
    validate_text_config(condition_config)
    validate_image_config(condition_config)

    with torch.no_grad():
        # Load tokenizer and text model based on config
        # Also get empty text representation
        text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
                                                             ['text_embed_model'], device=device)
        image_model, image_processor =  get_image_model_processor(condition_config['image_condition_config']['image_embed_model'], device = device)

    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device))
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config,
               text_tokenizer, text_model, image_model, image_processor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    args = parser.parse_args()
    infer(args)
