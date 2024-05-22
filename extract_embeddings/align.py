import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
import torch
import pandas as pd
import sys
sys.path.append('data')
from src.paths import MAIN_DIR_PATH, IM_FOLDER_PATH, OUTPUT_FOLDER, DATA_FOLDER
from os.path import join
import json
import pickle

model_name = "kakaobrain/align-base"


processor = AlignProcessor.from_pretrained(model_name)
model = AlignModel.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)



embedding_type = 'align_score_trade'
df = pd.read_csv(DATA_FOLDER / 'TRADE' / 'TRADE.csv')
ret_dict = {}

with torch.no_grad():
    for i in range(len(df)):
        im_dict = {}
        image = Image.open(IM_FOLDER_PATH / df.image_path[i]).convert('RGB')
        texts = df.loc[i, ['ar', 'distractor_1', 'distractor_2']].values.tolist()
        inputs = processor(text=texts, images=image, return_tensors="pt", padding='max_length', truncation=True)
        outputs = model(**inputs.to(device))
        im_dict['logits'] = outputs.logits_per_image.flatten().detach().cpu().numpy()
        im_dict['image_embedding'] = outputs['image_embeds'].detach().cpu().numpy()
        im_dict['text_embedding'] = outputs['text_embeds'].detach().cpu().numpy()
        im_dict['sentences'] = texts
        ret_dict[df.image_path[i]] = im_dict
        

config_dict = {'model_name': model_name,
            'model_component': ['logits_per_image', 'image_embeddings', 'text_embeddings'],
            'embedding_type': embedding_type}

align_outs = {'config' : config_dict, 'emb_dict': ret_dict}

model_name = model_name.split('/')[-1]
pickle.dump(align_outs, open(OUTPUT_FOLDER /'align' / f'{model_name}_{embedding_type}.pkl', "wb"))




