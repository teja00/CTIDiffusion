
# def validate_style_config(condition_config):
#     assert 'class_condition_config' in condition_config, \
#         "Class conditioning desired but class condition config missing"
#     assert 'num_classes' in condition_config['class_condition_config'], \
#         "num_class missing in class condition config"


def validate_text_config(condition_config):
    assert 'text_condition_config' in condition_config, \
        "Text conditioning desired but text condition config missing"
    assert 'text_embed_dim' in condition_config['text_condition_config'], \
        "text_embed_dim missing in text condition config"
    

def validate_image_config(condition_config):
   assert 'image_condition_config' in condition_config, \
        "Image conditioning desired but image condition config missing"
   assert 'image_embed_dim' in condition_config['image_condition_config'], \
        "image_embed_dim missing in image condition config"
    


# def validate_image_conditional_input(cond_input, x):
#     assert 'image' in cond_input, \
#         "Model initialized with image conditioning but cond_input has no image information"
#     assert cond_input['image'].shape[0] == x.shape[0], \
#         "Batch size mismatch of image condition and input"
#     assert cond_input['image'].shape[2] % x.shape[2] == 0, \
#         "Height/Width of image condition must be divisible by latent input"

def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value