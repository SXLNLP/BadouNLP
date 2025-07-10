# loader.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config_homework import Config

tokenizer = BertTokenizer.from_pretrained(Config["pretrain_model_path"])

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=Config["max_length"])
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

def load_dataset(path="文本分类练习.csv"):
    df = pd.read_csv(path)
    df.columns = ["label", "text"]
    df.dropna(inplace=True)
    return df
