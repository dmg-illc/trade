import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from os.path import join
import sys
sys.path.append('data')
from data.paths import MAIN_DIR_PATH, IM_FOLDER_PATH
import os
import pickle
import json
import sys
# sys.path.insert(0, os.path.abspath('/home/abavaresco/ads/ads_snellius/scripts'))
from utils import ModOutputs
from lavis.models import load_model_and_preprocess, load_model

dataset = 'ours' # can be either original or trade
subset = 'right' # can be right or wrong, used only on test set
OUTPUT_DIR = '/home/abavaresco/ads/ads_snellius/data/outputs'
IM_FOLDER_PATH = '/home/abavaresco/ads/images'
model_name = "albef_retrieval"
model_type = "flickr"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, vis_processors, txt_processors = load_model_and_preprocess(name = model_name,
                                                                  model_type = model_type, is_eval=True, device=device)
model.eval()


if dataset=='original':

    embedding_type = f'albef_score_test_{subset}'

    with open(join(MAIN_DIR_PATH, f'data/test_{subset}.json'), 'r') as my_file:
        ar_data=my_file.read()
    ar_dict = json.loads(ar_data)

    ret_dict = {}

    with torch.no_grad():
        for k in ar_dict:
            im_dict = {}
            
            text = [txt_processors['eval'](t) for t in ar_dict[k]]
            tokenized_text = model.tokenizer(text, padding="max_length", truncation=True, max_length=model.max_txt_len,
                                             return_tensors="pt").to(device)
            text_embeds = model.text_encoder.forward_text(tokenized_text).last_hidden_state
            text_features = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)

            img = Image.open(join(IM_FOLDER_PATH, k)).convert('RGB')
            image = vis_processors["eval"](img).unsqueeze(dim=0).to(device)
            image_embed = model.visual_encoder.forward_features(image)
            image_features = F.normalize(model.vision_proj(image_embed[:, 0, :]), dim=-1)


            im_dict['image_embedding'] = image_features.detach().cpu().numpy()
            im_dict['text_embedding'] = text_features.detach().cpu().numpy()
            im_dict['sentences'] = ar_dict[k]
            ret_dict[k] = im_dict
            

    config_dict = {'model_name': model_name,
                'temperature': model.temp.detach().cpu().numpy(),
                'embedding_type': embedding_type}

    albef_outs = ModOutputs(config=config_dict, outputs=ret_dict)

    model_name = model_name.split('/')[-1]
    pickle.dump(albef_outs, open(join(OUTPUT_DIR,'albef', f'{model_name}_{embedding_type}.pkl'), "wb"))

elif dataset=='trade':

    embedding_type = f'albef_score_trade'
    df = pd.read_csv(join(OUTPUT_DIR, 'aggregated_annotations.csv'))
    ret_dict = {}
    
    with torch.no_grad():
        for i in range(len(df)):
            im_dict = {}

            img = Image.open(join(IM_FOLDER_PATH, df.image_path[i])).convert('RGB')
            image = vis_processors["eval"](img).unsqueeze(dim=0).to(device)
            image_embed = model.visual_encoder.forward_features(image)
            image_features = F.normalize(model.vision_proj(image_embed[:, 0, :]), dim=-1)

            texts = df.loc[i, ['ar', 'distractor_1', 'distractor_2']].values.tolist()
            tokenized_text = model.tokenizer(texts, padding="max_length", truncation=True, max_length=model.max_txt_len,
                                             return_tensors="pt").to(device)
            text_embeds = model.text_encoder.forward_text(tokenized_text).last_hidden_state
            text_features = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)

            im_dict['image_embedding'] = image_features.detach().cpu().numpy()
            im_dict['text_embedding'] = text_features.detach().cpu().numpy()
            im_dict['sentences'] = texts
            ret_dict[df.image_path[i]] = im_dict
            
    
    config_dict = {'model_name': model_name,
                'temperature': model.temp.detach().cpu().numpy(),
                'embedding_type': embedding_type}

    albef_outs = ModOutputs(config=config_dict, outputs=ret_dict)

    model_name = model_name.split('/')[-1]
    pickle.dump(albef_outs, open(join(OUTPUT_DIR,'albef', f'{model_name}_{embedding_type}.pkl'), "wb"))

else:
    print('There was an error :(')


