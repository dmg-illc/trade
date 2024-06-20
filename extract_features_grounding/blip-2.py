from PIL import Image
import requests
import json
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import pandas as pd
from src.paths import OUTPUT_FOLDER, IM_FOLDER_PATH, DATA_FOLDER, TRADE_PATH
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset') # can be TRADE  or Pitt
args = parser.parse_args()
model_name = "Salesforce/blip2-opt-2.7b"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
processor = AutoProcessor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()


if args.dataset=='Pitt':

    embedding_type = f'blip_captions_wrong'

    with open(DATA_FOLDER / 'original_data' / 'test_wrong.json', 'r') as my_file:
        ar_data=my_file.read()
    ar_dict = json.loads(ar_data)


    ret_dict = {}

    with torch.no_grad():
        for k in ar_dict:
            im_dict = {}
            image = Image.open(IM_FOLDER_PATH / k).convert('RGB')
            inputs = processor(image, text='', return_tensors="pt")
            outputs = model(**inputs.to(device))
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            im_dict['caption'] = generated_text
            ret_dict[k] = im_dict
            

    config_dict = {'model_name': model_name,
                'model_component': 'generated_text',
                'embedding_type': embedding_type}

    blip_outs = {'config': config_dict, 'outputs': ret_dict}

    model_name = model_name.split('/')[-1]


elif args.dataset=='TRADE':

    embedding_type = 'blip_captions'
    df = pd.read_csv(TRADE_PATH)
    ret_dict = {}
    
    with torch.no_grad():
        for i in range(len(df)):
            im_dict = {}
            image = Image.open(IM_FOLDER_PATH / df.image_path[i]).convert('RGB')

            inputs = processor(image, text='', return_tensors="pt")
            outputs = model(**inputs.to(device))
            generated_ids = model.generate(**inputs, max_new_tokens=30)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            im_dict['caption'] = generated_text
            ret_dict[df.image_path[i]] = im_dict
            
            
    
    config_dict = {'model_name': model_name,
                'model_component': 'generated_text',
                'embedding_type': embedding_type}

    blip_outs = {'config': config_dict, 'outputs': ret_dict}

    model_name = model_name.split('/')[-1]


else:
    print('Wrong dataset argument!')

pickle.dump(blip_outs, open(OUTPUT_FOLDER / 'blip' / f'{model_name}_{embedding_type}.pkl'), "wb")





