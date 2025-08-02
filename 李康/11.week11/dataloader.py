import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BertModel.from_pretrained("bert-base-chinese")
model = model.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
max_length = 512
"""
数据生成: 使用标题作为x，内容作为y
"""

class NewsData(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        seq = self.data[item]
        input_seq = f"{seq['title']}"
        output_seq = f"{seq['content']}"
        return input_seq, output_seq

def collote_fn(batch_datas):
    input_sentence = []
    output_sentence = []
    for (input_seq, output_seq) in batch_datas:
        input_sentence.append(input_seq)
        output_sentence.append(output_seq)
    batch_data = tokenizer(
        input_sentence,
        text_target=output_sentence,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(batch_data['labels'])
    return batch_data