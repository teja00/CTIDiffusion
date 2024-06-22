import glob
import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class IAMDataset(Dataset):
    r"""
    IAM dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """
    
    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='png',condition_config=None):
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        
        self.condition_types = [] if condition_config is None else condition_config['condition_types']
        
            
        self.images, self.texts = self.load_images(im_path)
        
    
    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        file = open('/kaggle/input/iam-handwriting-word-database/words_new.txt', "r")
        file_content = file.readlines()
        texts = []
        
        
        for content in (file_content):
            split_content = content.split(' ')
            
            if 'text' in self.condition_types:
                texts.append(split_content[-1])
            if 'image' in self.condition_types:
                image_name = split_content[0]
                image_name_folder_1 = image_name.split('-')[0]
                image_name_folder_2 = image_name_folder_1 + '-' + image_name.split('-')[1] 
                image_final_folder = image_name_folder_1 + '/' + image_name_folder_2 + '/' + image_name
                ims.append(os.path.join(im_path, 'words/{}.{}'.format(image_final_folder, self.im_ext)))
        if 'text' in self.condition_types:
            assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"
        print('Found {} images'.format(len(ims)))
        print('Found {} captions'.format(len(texts)))
        return ims, texts
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = self.texts[index]

        #######################################
        
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(),
        ])(im)
        im.close()
    
        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1

        if 'image' in self.condition_types:
            cond_inputs['image'] = im
        if len(self.condition_types) == 0:
            return im_tensor
        else:
            return im_tensor, cond_inputs
