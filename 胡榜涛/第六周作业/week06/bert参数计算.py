import torch
import math
import numpy as np
from transformers import BertModel
bert = BertModel.from_pretrained(r"D:\AI课程学习\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()

def parameter_number_true(state_dict):
    total_params = 0
    for param_tensor in state_dict.values():
        # 计算当前参数的元素数量（形状各维度的乘积）
        param_count = param_tensor.numel()
        total_params += param_count

    return total_params

def parameter_calcuation(hidensize,vocab_size,max_length,layers_num):
    token_embeddings = vocab_size*hidensize
    segment_embeddings = 2*hidensize
    position_embeddings = max_length*hidensize
    attention_q=hidensize*hidensize+hidensize
    attention_k=hidensize*hidensize+hidensize
    attention_v = hidensize*hidensize+hidensize
    Layer_Normalization_size=2*hidensize*2*2
    out_linear=hidensize*hidensize+hidensize
    feed_forward_1=hidensize*4*hidensize+4*hidensize
    feed_forward_2=4*hidensize*hidensize+hidensize
    sigle_total_parameters_size=attention_q+attention_k+attention_v+out_linear+feed_forward_1+feed_forward_2+Layer_Normalization_size
    total_parameters_size=token_embeddings+segment_embeddings+position_embeddings+layers_num*sigle_total_parameters_size
    return total_parameters_size

if __name__ == '__main__':
    hiddensize = 768
    vocab_size = 21128
    max_length = 512
    layers_num = 12
    total_parameters_size = parameter_calcuation(hiddensize,vocab_size,max_length,layers_num)
    print(f'手动计算参数量为{total_parameters_size/ 1e6:.2f}M(百万)')
    true_parameters_size = parameter_number_true(state_dict)
    print(f'正确参数量为:{true_parameters_size/ 1e6:.2f}M(百万)')
