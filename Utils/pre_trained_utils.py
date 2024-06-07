import torch
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel, AutoProcessor, CLIPModel

# TODO we need to define the Text to embedding models
# TODO we need to define the pretrained image embedding models
# TODO we need to define the pretrained style embedding models [this would essentially mean styleid to embedding similar to text model the stylegan2 model]


def get_tokenizer_and_model(model_type, device, eval_mode=True):
    if model_type == 'bert':
        text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    if model_type == 'clip':
        text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
    if eval_mode:
        text_model.eval()
    return text_tokenizer, text_model

def get_image_mode_processor(model_type, device, eval_model=True):
    if model_type == 'clip':
        image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if eval_model == True:
        image_model.eval()
    return image_model, image_processor
    

# def get_text_representation(text, text_tokenizer, text_model, device,
#                             truncation=True,
#                             padding='max_length',
#                             max_length=77):
#     token_output = text_tokenizer(text,
#                                   truncation=truncation,
#                                   padding=padding,
#                                   return_attention_mask=True,
#                                   max_length=max_length)
#     indexed_tokens = token_output['input_ids']
#     att_masks = token_output['attention_mask']
#     tokens_tensor = torch.tensor(indexed_tokens).to(device)
#     mask_tensor = torch.tensor(att_masks).to(device)
#     text_embed = text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state
#     return text_embed

def get_text_representation(text, text_tokenizer, text_model, device,
                            truncation=True,
                            padding='max_length',
                            max_length=77):
    token_output = text_tokenizer(text,
                                  truncation=truncation,
                                  padding=padding,
                                  return_attention_mask=True,
                                  max_length=max_length,
                                  return_tensors='pt')  # Return PyTorch tensors
    tokens_tensor = token_output['input_ids'].to(device)
    mask_tensor = token_output['attention_mask'].to(device)
    
    with torch.no_grad():  # Disable gradient calculation
        text_embed = text_model(input_ids=tokens_tensor, attention_mask=mask_tensor).last_hidden_state
    
    return text_embed

def get_image_representation(image, image_model, image_processor, device):
    token_input = inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = image_model.get_image_features(**token_input)

    return image_features
    