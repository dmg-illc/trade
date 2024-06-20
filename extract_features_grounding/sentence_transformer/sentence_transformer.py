from bert_utils import *
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import pickle
import json
import argparse
from src.paths import OUTPUT_FOLDER, DATA_FOLDER

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset') # can be TRADE  or Pitt
parser.add_argument('-s', '--subset') # can corr_ar, wrong_ar or ocr_text
args = parser.parse_args()

# OUTPUT_DIR = '/home/abavaresco/ads/ads_snellius/data/outputs'
# SIM_TEXT_PATH = '/home/abavaresco/ads/ads_snellius/data/outputs/text_sim.csv'
# ARS_PATH = '/home/abavaresco/ads/ads_snellius/data/outputs/test_wrong.json'
out_dim = 768
model_name = "all-mpnet-base-v2"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = SentenceTransformer(model_name)

batch_size = 200

if args.dataset=='TRADE':
    df = pd.read_csv(DATA_FOLDER / 'original_data' / 'dist_w_ocr.csv')

    if args.subset == 'wrong_ar':
        sent_list = df.distractor_1.values.tolist() + df.distractor_2.values.tolist()
        embedding_type = 'wrong_ar_trade'
        
    elif args.subset == 'corr_ar':
        sent_list = df.ar
        embedding_type = 'corr_ar_trade'

    elif args.subset == 'ocr_text':
        sent_list = df.text
        embedding_type = 'ocr_text_trade'
    

elif args.dataset=='Pitt':

    if args.subset == 'wrong_ar' or args.subset == 'corr_arr':

        if args.subset == 'wrong_ar':
            ARS_PATH = DATA_FOLDER / 'original_data' / 'test_wrong.json'
            embedding_type = 'wrong_ar'

        elif args.subset == 'corr_arr':
            ARS_PATH = DATA_FOLDER / 'original_data' / 'test_right.json'
            embedding_type = 'corr_ar'

        with open(ARS_PATH, 'r') as myfile:
            data=myfile.read()

        data_dict = json.loads(data)
        sent_list = []

        for k in data_dict:
            answers = data_dict[k]
            for ans in answers:
                sent_list.append(ans)

    elif args.subset == 'ocr_text':
        embedding_type = 'oce_text'
        ocr_df = pd.read_csv(DATA_FOLDER / 'original_data' / 'text_sim.csv')
        sent_list = ocr_df.text[~ocr_df.text.isnull()]



embeddings = model.encode(sent_list, batch_size = batch_size)
emb_dict = {sent_list[i]: embeddings[i] for i in range(len(sent_list))}

config_dict = {'model_name': model_name,
               'model_component': '',
               'embedding_type': embedding_type}
ret_dict = {'config': config_dict, 'emb_dict': emb_dict}


pickle.dump(ret_dict, open(OUTPUT_FOLDER / f'{model_name}_{embedding_type}.pkl', "wb"))



