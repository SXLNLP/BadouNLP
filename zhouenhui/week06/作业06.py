import torch
import math
import numpy as np
from transformers import BertModel


# embedding嵌入层
class diy_Calculator:
    def __init__(self):
        self.vocab_size = 21128
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.intermediate_size = 3072

    # 计算embedding层
    def calculate_embedding(self):
        calculate_embedding_count = (self.vocab_size + self.max_position_embeddings + self.type_vocab_size) * self.hidden_size
        print(f"Total embedding parameters: {calculate_embedding_count}")
        return calculate_embedding_count

    # 计算transformer Encoder层：
    def self_attention(self):
        self_attention_parameters = 4 * self.hidden_size * (self.hidden_size / self.num_attention_heads) * self.num_attention_heads * 12
        print(f"Total attention parameters:{self_attention_parameters}")
        return self_attention_parameters

    def feed_forward(self):
        feed_forward_parameters = 2 * self.hidden_size * self.intermediate_size * 12
        print(f"Total ff parameters:{feed_forward_parameters}")
        return feed_forward_parameters

    def LayerNorm(self):
        LayerNorm_parameters = 2 * self.hidden_size * 12
        print(f"Total LayerNorm parameters:{LayerNorm_parameters}")
        return LayerNorm_parameters

    def pooler(self):
        pooler_parameters = self.hidden_size * self.hidden_size
        print(f"Total pooler parameters:{pooler_parameters}")
        return pooler_parameters

    def calculate_total(self):
        ca = self.calculate_embedding()
        sa = self.self_attention()
        ff = self.feed_forward()
        ln = self.LayerNorm()
        pl = self.pooler()
        total = ca + sa + ff + ln + pl
        print(f"\nTotal parameters:{int(total)} 个")
        return int(total)

    def total_bytes(self):
        total = self.calculate_total()
        total_bytes = total * 4 # 每个beat 占 4个byte；
        total_size = total_bytes / 1024 / 1024
        print(f"Total bytes:{total_bytes} b")
        print(f"Total size:{int(total_size)} MB")


calculator = diy_Calculator()
calculator.total_bytes()
