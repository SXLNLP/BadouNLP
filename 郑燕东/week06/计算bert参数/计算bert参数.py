import torch
import math
import numpy as np
from transformers import BertModel

#计算bert有多少参数，需要多大显存
#embedding嵌入层
class diy_Calculator:
    def __init__(self):
        self.vocab_size=21128
        self.max_position_embeddings=512
        self.type_vocab_size=2
        self.hidden_size=768
        self.num_attention_heads = 12
        self.intermediate_size = 3072
    #计算embedding层
    def calculate_embedding(self):
        x=(self.vocab_size+ self.max_position_embeddings +self.type_vocab_size)*self.hidden_size
        print(f"Total embedding parameters: {x}")
        return x
#计算transformer Encoder层：
    def self_attention(self):
        y=4*self.hidden_size*(self.hidden_size/self.num_attention_heads)*self.num_attention_heads*12
        print(f"Total attention parameters:{y}")
        return y
    def feed_forward(self):
        y1=2*self.hidden_size*self.intermediate_size*12
        print(f"Total ff parameters:{y1}")
        return y1
    def LayerNorm(self):
        y2=2*self.hidden_size*12
        print(f"Total LayerNorm parameters:{y2}")
        return y2
    def pooler(self):
        z=self.hidden_size*self.hidden_size
        print(f"Total pooler parameters:{z}")
        return z
    def calculate_total(self):
        embedding = self.calculate_embedding()
        attention = self.self_attention()
        ff = self.feed_forward()
        norm = self.LayerNorm()
        pooler = self.pooler()
        total=embedding + attention + ff + norm + pooler
        print(f"\nTotal parameters:{total}")
        return total
    def total_bytes(self):
        total=self.calculate_total()
        total_bytes = str(total*4)[:3]
        print(f"Total bytes:{total_bytes}MB,推理时可使用8G显存，实际训练需要12G显存")
calculator = diy_Calculator()
calculator.total_bytes()
