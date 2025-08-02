# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    # 输入输出转化成序列
    def prepare_data(self, title, content):
        # 你好吗 sep 120
        input_max_length = self.config["input_max_length"]
        content_seq = self.tokenizer.encode(content, add_special_tokens=False)
        title_seq = self.tokenizer.encode(title, add_special_tokens=False)

        input_seq = [self.tokenizer.cls_token_id] + content_seq + [self.tokenizer.sep_token_id] + title_seq + [
            self.tokenizer.sep_token_id]
        gold = len(content_seq) * [-100] + [-100] + title_seq + [self.tokenizer.sep_token_id] + [-100]

        input_seq = input_seq[:input_max_length] + [0] * (input_max_length - len(input_seq))
        gold = gold[:input_max_length] + [0] * (input_max_length - len(gold))

        # print(content)
        # print(len(input_seq))
        # print(title)
        # print(len(gold))
        # encoded_inputs = self.tokenizer.encode(content, add_special_tokens=False, truncation=True,
        #                                        max_length=input_max_length,
        #                                        )
        # # 我很好
        # encoded_outputs = self.tokenizer.encode(title, add_special_tokens=False, truncation=True,
        #                                         max_length=output_max_length,
        #                                         )
        #
        # input_seq = encoded_inputs + [self.tokenizer.sep_token_id] + encoded_outputs
        # input_seq = input_seq[:input_max_length + output_max_length]
        # input_seq += [self.tokenizer.pad_token_id] * (
        #         input_max_length + output_max_length - len(input_seq))
        # # 9  + 10
        # gold = [-100] * len(encoded_inputs) + encoded_outputs
        # gold += [-100] * (input_max_length + output_max_length - len(gold))
        self.data.append([torch.LongTensor(input_seq),
                          torch.LongTensor(gold)])

        # [6484, 3625, 1525, 2399, 1158, 4989, 102, 6484, 3625, 2768, 4989, 754, 8368, 2399, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # [-100, -100, -100, -100, -100, -100, 6484, 3625, 2768, 4989, 754, 8368, 2399, 102, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
        #  -100, -100, -100, -100, -100]
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


def build_causal_attention_mask(sep_token_id=102):
    # 原始序列         你     好    吗     ?    我    很     好
    # input_seq = [101, 3790, 4497, 7452, 6666, 102]
    # output_seq = [101, 3790, 4497, 7452, 102]
    # gold = [101, 3790, 4497, 7452, 102]

    input_seq = [101, 3790, 4497]
    output_seq = [7452, 102]
    gold = [7452, 102]

    label = gold[:2]
    label = [-100] * (3 - 2) + label
    print(label)
    # 转换为 tensor
    input_seq = torch.tensor(input_seq, dtype=torch.long)
    output_seq = torch.tensor(output_seq, dtype=torch.long)
    gold = torch.tensor(gold, dtype=torch.long)
    input_seq_len = len(input_seq)
    output_seq_len = len(output_seq)
    gold_seq_len = len(gold)
    gold += torch.tensor([-100]) * (input_seq_len - gold_seq_len)
    print(gold)

    left1_mask = torch.ones(input_seq_len, input_seq_len, dtype=torch.float32)
    print(left1_mask)
    right1_mask = torch.zeros(input_seq_len, output_seq_len, dtype=torch.float32)
    print(right1_mask)
    left2_mask = torch.ones(output_seq_len, input_seq_len, dtype=torch.float32)
    print(left2_mask)
    right2_mask = torch.tril(torch.ones(output_seq_len, output_seq_len, dtype=torch.float32))
    print(right2_mask)

    concatenated_mask1 = torch.cat((left1_mask, right1_mask), dim=1)
    concatenated_mask2 = torch.cat((left2_mask, right2_mask), dim=1)
    concatenated_mask = torch.cat((concatenated_mask1, concatenated_mask2), dim=0)
    print(concatenated_mask)
    #
    # print(input_ids)
    # # 找出所有 [SEP] 的位置
    # sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    #
    # if len(sep_positions) < 2:
    #     raise ValueError("未找到两个 [SEP] 标记")
    #
    # seq_len = len(input_ids)
    # mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    #
    # # 输入部分：从开始到第一个 [SEP] 后一个位置
    # input_end = sep_positions[0].item() + 1
    #
    # # 输出部分：从第二个 [SEP] 开始的位置
    # output_start = sep_positions[1].item()
    #
    # for i in range(seq_len):
    #     if i < input_end:
    #         # 输入部分可以看到整个输入区域
    #         mask[i, :input_end] = 0
    #     else:
    #         # 输出部分只能看到输入 + 已生成的输出（三角矩阵）
    #         mask[i, :i + 1] = 0  # 下三角（含对角线）
    #     # 其他位置默认为 -inf（不可见）
    #     mask[i, mask[i] == 0] = 1
    #     mask[i, mask[i] != 1] = float('-inf')
    #     mask[i, mask[i] == 1] = 0
    # print(mask)
    return


if __name__ == "__main__":
    from config import Config

    # dl = load_data(Config["train_data_path"], Config, 1)
    # for batch in dl:
    #     print(batch)
    #     break
    build_causal_attention_mask()
