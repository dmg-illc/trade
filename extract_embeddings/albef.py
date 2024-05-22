import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from os.path import join
from src.paths import MAIN_DIR_PATH, IM_FOLDER_PATH, OUTPUT_FOLDER, DATA_FOLDER
import os
import pickle
import json
from lavis.models import load_model_and_preprocess, load_model

model_name = "albef_retrieval"
model_type = "flickr"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, vis_processors, txt_processors = load_model_and_preprocess(name = model_name,
                                                                  model_type = model_type, is_eval=True, device=device)
model.eval()


embedding_type = 'albef_score_trade'
df = pd.read_csv(DATA_FOLDER / 'TRADE' / 'TRADE.csv')
ret_dict = {}

with torch.no_grad():
    for i in range(len(df)):
        im_dict = {}

        img = Image.open(IM_FOLDER_PATH / df.image_path[i]).convert('RGB')
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

albef_outs = {'config': config_dict, 'emb_dict':ret_dict}

model_name = model_name.split('/')[-1]
pickle.dump(albef_outs, open(OUTPUT_FOLDER / 'albef', f'{model_name}_{embedding_type}.pkl', "wb"))



