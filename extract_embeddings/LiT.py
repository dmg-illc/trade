# !pip install -q git+https://github.com/google-research/vision_transformer

import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
from os.path import join
import torch
from vit_jax import models
import pandas as pd
import json
from src.paths import MAIN_DIR_PATH, IM_FOLDER_PATH

model_name = 'LiT-L16L'

lit_model = models.get_model(model_name)
# Loading the variables from cloud can take a while the first time...
lit_variables = lit_model.load_variables()
# Creating tokens from freeform text (see next section).
tokenizer = lit_model.get_tokenizer()
# Resizing images & converting value range to -1..1 (see next section).
image_preprocessing = lit_model.get_image_preprocessing()
# Preprocessing op for use in tfds pipeline (see last section).
pp = lit_model.get_pp()


class pitts_ads_dataset():
    def __init__(self, df, img_folder):
        # Initialize image paths and corresponding texts
        image_paths = [os.path.join(img_folder, path) for path in list(df.image_path)]
        self.images = [Image.open(path).convert('RGB') for path in image_paths]


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        img = self.images[idx]

        return img
    

df = pd.read_csv(join(MAIN_DIR_PATH, 'data/TRADE.csv'))
dataset = pitts_ads_dataset(df, IM_FOLDER_PATH)

processed_images = image_preprocessing(dataset.images)
zimg, _, _ = lit_model.apply(lit_variables, images=processed_images)

text = list(df.ar)
tokens = tokenizer(text)
_, zars, _ = lit_model.apply(lit_variables, tokens=tokens)

text = list(df.distractor_1)
tokens = tokenizer(text)
_, zd1, _ = lit_model.apply(lit_variables, tokens=tokens)

text = list(df.distractor_2)
tokens = tokenizer(text)
_, zd2, _ = lit_model.apply(lit_variables, tokens=tokens)

torch.save({'images' : torch.tensor(np.array(zimg)),
            'ar': torch.tensor(np.array(zars)),
            'dist1': torch.tensor(np.array(zd1)),
            'dist2': torch.tensor(np.array(zd2)),
            'model_checkpoint': 'LiT-L16L',
            'temperature': torch.tensor(np.array(lit_variables['params']['t']))},
           join(MAIN_DIR_PATH,'data/model_outputs/lit_outputs_trade'))