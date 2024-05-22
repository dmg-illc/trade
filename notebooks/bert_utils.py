import numpy as np


class SentenceEmbeddings:
    def __init__(self, config: dict, outputs: dict):
        '''
            config is a dict with the following keys:
                1. model_name --> e.g. bert-base-uncased
                2. model_component --> e.g. pooled_output
                3. logit_type --> oct_text, corr_ar, wrong_ar
                3. Whatever else you think would be useful to save
            
            embeddings is a dict where the keys are the concept
            names for text embeddings and the image names (e.g. ability_1) for the images. 
            The values are the embeddings (in np array format) of the text/image


        '''
        self.config = config
        self.out_dict = outputs
        self.images = list(self.out_dict.keys())


    def __repr__(self):
        configstr = "\n\t".join([f"{k}: {v}" for k, v in self.config.items()])
        return f"SentenceEmbeddings()\n\t{configstr}"

    def __call__(self, image_name):
        return self.out_dict[image_name]


