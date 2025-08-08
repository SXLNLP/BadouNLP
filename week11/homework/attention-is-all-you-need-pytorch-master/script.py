
import torch
from transformer.Models import Transformer


transformer = Transformer(
        1000,
        1000,
        src_pad_idx=0,
        trg_pad_idx=0)

x = torch.LongTensor([[3,4,56,7,8,6]])
y = torch.LongTensor([[3,2,5,1,0,0]])
s = transformer(x, y)
print(s,s.shape)