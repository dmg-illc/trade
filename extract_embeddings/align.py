import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel
import torch
import pandas as pd
import sys
sys.path.append('data')
from data.paths import MAIN_DIR_PATH, IM_FOLDER_PATH
from os.path import join
import json
import pickle
from utils import ModOutputs

dataset = 'ours' # can be either original or trade
subset = 'wrong' # used only on test set
OUTPUT_DIR = join(MAIN_DIR_PATH, 'data/model_outputs')
model_name = "kakaobrain/align-base"


processor = AlignProcessor.from_pretrained(model_name)
model = AlignModel.from_pretrained(model_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

if dataset=='original':

    embedding_type = f'align_score_test_{subset}'

    with open(join(MAIN_DIR_PATH, f'data/test_{subset}.json'), 'r') as my_file:
        ar_data=my_file.read()
    ar_dict = json.loads(ar_data)


    ret_dict = {}

    with torch.no_grad():
        for k in ar_dict:
            im_dict = {}
            image = Image.open(join(IM_FOLDER_PATH, k)).convert('RGB')
            inputs = processor(text=ar_dict[k], images=image, return_tensors="pt", padding='max_length', truncation=True)
            outputs = model(**inputs.to(device))
            im_dict['logits'] = outputs.logits_per_image.flatten().detach().cpu().numpy()
            im_dict['image_embedding'] = outputs['image_embeds'].detach().cpu().numpy()
            im_dict['text_embedding'] = outputs['image_embeds'].detach().cpu().numpy()
            im_dict['sentences'] = ar_dict[k]
            ret_dict[k] = im_dict

    config_dict = {'model_name': model_name,
                'model_component': 'logits_per_image',
                'embedding_type': embedding_type}

    align_outs = ModOutputs(config=config_dict, outputs=ret_dict)

    model_name = model_name.split('/')[-1]
    pickle.dump(align_outs, open(join(OUTPUT_DIR,'align', f'{model_name}_{embedding_type}.pkl'), "wb"))

elif dataset=='trade':

    embedding_type = f'align_score_trade'
    df = pd.read_csv(join(OUTPUT_DIR, 'aggregated_annotations.csv'))
    ret_dict = {}
    
    with torch.no_grad():
        for i in range(len(df)):
            im_dict = {}
            image = Image.open(join(IM_FOLDER_PATH, df.image_path[i])).convert('RGB')
            texts = df.loc[i, ['ar', 'distractor_1', 'distractor_2']].values.tolist()
            inputs = processor(text=texts, images=image, return_tensors="pt", padding='max_length', truncation=True)
            outputs = model(**inputs.to(device))
            im_dict['logits'] = outputs.logits_per_image.flatten().detach().cpu().numpy()
            im_dict['image_embedding'] = outputs['image_embeds'].detach().cpu().numpy()
            im_dict['text_embedding'] = outputs['image_embeds'].detach().cpu().numpy()
            im_dict['sentences'] = texts
            ret_dict[df.image_path[i]] = im_dict
            
    
    config_dict = {'model_name': model_name,
                'model_component': ['logits_per_image', 'image_embeddings', 'text_embeddings'],
                'embedding_type': embedding_type}

    align_outs = ModOutputs(config=config_dict, outputs=ret_dict)

    model_name = model_name.split('/')[-1]
    pickle.dump(align_outs, open(join(OUTPUT_DIR,'align', f'{model_name}_{embedding_type}.pkl'), "wb"))

else:
    print('There was an error :(')



