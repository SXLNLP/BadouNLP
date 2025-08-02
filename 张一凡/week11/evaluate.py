# -*- coding: utf-8 -*-
import torch
import collections
import io
import json
import six
import sys
import argparse
from loader import load_data
from collections import defaultdict, OrderedDict

from transformer.Translator import Translator

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, logger, shuffle=False)
        self.reverse_vocab = dict([(y, x) for x, y in self.valid_data.dataset.vocab.items()])
        self.translator = Translator(self.model,
                                     config["beam_size"],
                                     config["output_max_length"],
                                     config["pad_idx"],
                                     config["pad_idx"],
                                     config["start_idx"],
                                     config["end_idx"])

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.model.cpu()

        for index, batch_data in enumerate(self.valid_data):
            input_seqs, target_seqs, gold = batch_data

            # 只评估第一个样本作为示例
            input_seq = input_seqs[0].unsqueeze(0)
            target_seq = target_seqs[0].unsqueeze(0)

            # 生成回答
            generate = self.translator.translate_sentence(input_seq)

            # 解码输入、目标和生成结果
            input_text = self.decode_seq(input_seq[0])
            target_text = self.decode_seq(target_seq[0])
            generate_text = self.decode_seq(generate)

            print(f"\n样本 {index + 1}:")
            print("文章内容:", input_text.replace("[PAD]", "").strip())
            print("真实标题:", target_text.replace("[PAD]", "").strip())
            print("生成标题:", generate_text.replace("[PAD]", "").strip())

            # 只展示前几个样本
            if index >= 2:
                break

        return

    def decode_seq(self, seq):
        return "".join([self.reverse_vocab.get(int(idx), "[UNK]") for idx in seq])


if __name__ == "__main__":
    label = [2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2]

    print([(i, l) for i, l in enumerate(label)])
