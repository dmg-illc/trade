# TRADE
 
This is the GitHub Repository for the paper [_Don’t Buy it! Reassessing the Ad Understanding Abilities
of Contrastive Multimodal Models_](https://arxiv.org/abs/2405.20846) (A. Bavaresco, A. Testoni, R. Fernández), to appear at ACL 2024.

TRADE and TRADE-control are publicly available at: https://zenodo.org/records/11355892

## Pitt Ads Images

The complete set of images included in the Pitt Ads Dataset is publicly available [here](https://people.cs.pitt.edu/~mzhang/image_ads/) by courtesy of the authors of [Automatic Understanding of Image and Video Advertisements](https://openaccess.thecvf.com/content_cvpr_2017/papers/Hussain_Automatic_Understanding_of_CVPR_2017_paper.pdf).

To run the code in this repo, please download all images from the Pitt Ads Dataset and place them in `data/images` (without subfolders).

## Model performance on TRADE

The code to reproduce the performance of CLIP, LiT, ALBEF and ALIGN on TRADE is located in `notebooks/performance_on_trade.ipynb`. The model outputs used in this notebook can be downloaded [here](https://surfdrive.surf.nl/files/index.php/s/M8ojV48yirTTJ6X). The code we used to extract these model outputs is available in `extract embeddings`.

## Grounding scores

The code to compute the grounding scores for positive and negative ad explanations is provided in `notebooks/grounding_scores.ipynb`. Again, the outputs used in the notebooks can be downloaded [here](https://surfdrive.surf.nl/files/index.php/s/M8ojV48yirTTJ6X), whereas the code used to yield them is in `extract_features_grounding`.

## Human performance on TRADE

The code to reproduce our results on human performance on TRADE is provided in `notebooks/human_performance.ipynb`.

## Citation

If you are using TRADE, please cite our paper:

```
@article{bavaresco2024don,
  title={Don't Buy it! Reassessing the Ad Understanding Abilities of Contrastive Multimodal Models},
  author={Bavaresco, Anna and Testoni, Alberto and Fern{\'a}ndez, Raquel},
  journal={arXiv preprint arXiv:2405.20846},
  year={2024}
}
```

 
