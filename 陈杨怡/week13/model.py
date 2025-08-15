import torch.nn as nn
from transformers import AutoModelForTokenClassification
from config import Config

class NERModel:
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], num_labels=Config["class_num"])

    def get_model(self):
        return self.model
