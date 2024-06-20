import sys
import torch
import os
import pickle
from src.paths import TRADE_PATH, OUTPUT_FOLDER, IM_FOLDER_PATH, DATA_FOLDER
import json

sys.path.insert(0, os.path.abspath('./detectron2'))
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import argparse

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pandas as pd
import requests
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset') # can be TRADE  or Pitt
args = parser.parse_args()



cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


out_dict = {}

if args.dataset == 'TRADE':
    df = pd.read_csv(TRADE_PATH)

    with torch.no_grad():
        for i in range(len(df)):
            path = df.image_path[i]
            im_name = path.split('/')[-1]
            try:
                pil_im = Image.open(IM_FOLDER_PATH / path).convert('RGB')
                im = np.array(pil_im)[:, :, ::-1]
                outputs = predictor(im)
                ret_outputs = {}
                ret_outputs['scores'] = outputs["instances"].scores.detach().cpu()
                ret_outputs['pred_classes'] = outputs['instances'].pred_classes.detach().cpu()
                out_dict[im_name] = ret_outputs
            except:
                print(path)
                continue
    pickle.dump(out_dict, open(os.path.join(OUTPUT_FOLDER,'det2', 'trade_slurm_mask_rcnn_R_50_FPN_3x.pkl'), "wb"))

elif args.dataset == 'Pitt':
    with open(DATA_FOLDER / 'original_data' / 'test_wrong.json', 'r') as my_file:
        ar_data=my_file.read()
    ar_dict = json.loads(ar_data)

    with torch.no_grad():
        for k in ar_dict:   
            path = k
            im_name = path.split('/')[-1]
            try:
                pil_im = Image.open(IM_FOLDER_PATH / path).convert('RGB')
                im = np.array(pil_im)[:, :, ::-1]
                outputs = predictor(im)
                ret_outputs = {}
                ret_outputs['scores'] = outputs["instances"].scores.detach().cpu()
                ret_outputs['pred_classes'] = outputs['instances'].pred_classes.detach().cpu()
                out_dict[im_name] = ret_outputs
            except:
                print(path)
                continue
    pickle.dump(out_dict, open(os.path.join(OUTPUT_FOLDER,'det2', 'short_mask_rcnn_R_50_FPN_3x.pkl'), "wb"))

else:
    print('Invalid dataset argument!')




