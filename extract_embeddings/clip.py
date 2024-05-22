from PIL import Image
import requests
import json
from os.path import join
from src.paths import MAIN_DIR_PATH, IM_FOLDER_PATH, OUTPUT_FOLDER, DATA_FOLDER
from transformers import CLIPProcessor, CLIPModel
import torch
import pandas as pd
# from bert_utils import *
import pickle

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset') # can be TRADE  or Pitt
parser.add_argument('-s', '--subset') # can be either right or wrong
args = parser.parse_args()

model_name = "openai/clip-vit-large-patch14-336"



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained(model_name).to(device)
model.eval()
processor = CLIPProcessor.from_pretrained(model_name)

if args.dataset=='Pitt':

    embedding_type = f'clip_score_test_{args.subset}'

    with open(join(OUTPUT_FOLDER, f'test_{args.subset}.json'), 'r') as my_file:
        ar_data=my_file.read()
    ar_dict = json.loads(ar_data)


    ret_dict = {}

    with torch.no_grad():
        for k in ar_dict:
            im_dict = {}
            image = Image.open(IM_FOLDER_PATH / k).convert('RGB')
            inputs = processor(text=ar_dict[k], images=image, return_tensors="pt", padding='max_length', truncation=True)
            outputs = model(**inputs.to(device))
            logits = outputs.logits_per_image.flatten().detach().cpu().numpy()
            im_dict['logits'] = logits
            im_dict['image_embedding'] = outputs['image_embeds'].detach().cpu().numpy()
            im_dict['text_embedding'] = outputs['text_embeds'].detach().cpu().numpy()
            im_dict['sentences'] = ar_dict[k]
            ret_dict[k] = im_dict

    config_dict = {'model_name': model_name,
                'model_component': 'logits_per_image',
                'embedding_type': embedding_type}

    clip_outs = {'config':config_dict, 'emb_dict':ret_dict}

    model_name = model_name.split('/')[-1]
    pickle.dump(clip_outs, open(join(OUTPUT_FOLDER,'clip', f'{model_name}_{embedding_type}.pkl'), "wb"))

elif args.dataset=='TRADE':

    embedding_type = f'clip_score_our_dataset'
    df = pd.read_csv(DATA_FOLDER / 'TRADE' / 'TRADE.csv')
    ret_dict = {}
    
    with torch.no_grad():
        for i in range(len(df)):
            im_dict = {}
            image = Image.open(IM_FOLDER_PATH / df.image_path[i]).convert('RGB')
            texts = df.loc[i, ['ar', 'distractor_1', 'distractor_2']].values.tolist()
            inputs = processor(text=texts, images=image, return_tensors="pt", padding='max_length', truncation=True)
            outputs = model(**inputs.to(device))
            logits = outputs.logits_per_image.flatten().detach().cpu().numpy()
            im_dict['logits'] = logits
            im_dict['image_embedding'] = outputs['image_embeds'].detach().cpu().numpy()
            im_dict['text_embedding'] = outputs['text_embeds'].detach().cpu().numpy()
            im_dict['sentences'] = texts
            ret_dict[df.image_path[i]] = im_dict
            
    
    config_dict = {'model_name': model_name,
                'model_component': ['logits_per_image', 'image_embeddings', 'text_embeddings'],
                'embedding_type': embedding_type}

    clip_outs = {'config' : config_dict, 'emb_dict': ret_dict}
    model_name = model_name.split('/')[-1]
    pickle.dump(clip_outs, open(OUTPUT_FOLDER / 'clip' / f'{model_name}_{embedding_type}.pkl', "wb"))

else:
    print('There was an error :(')





