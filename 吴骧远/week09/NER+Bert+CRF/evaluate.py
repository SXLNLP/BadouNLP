# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from transformers import BertTokenizer

"""
模型效果测试 - 使用BertTokenizer实现标签对齐
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[
                        index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, attention_mask, labels = batch_data#有三个输入
            with torch.no_grad():
                pred_results = self.model(input_ids, attention_mask)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results, sentences, input_ids, attention_mask)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences, input_ids, attention_mask):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for i, (true_label, pred_label, sentence) in enumerate(zip(labels, pred_results, sentences)):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            # 将token大小的标签转换为单个字符大小的标签
            input_id = input_ids[i].cpu().detach().tolist()
            attn_mask = attention_mask[i].cpu().detach().tolist()

            # 获取有效的token（排除padding, CLS, SEP）
            valid_length = sum(attn_mask)
            true_label_chars = self.convert_token_labels_to_char_labels(
                sentence, input_id[:valid_length], true_label[:valid_length]
            )
            pred_label_chars = self.convert_token_labels_to_char_labels(
                sentence, input_id[:valid_length], pred_label[:valid_length]
            )

            true_entities = self.decode(sentence, true_label_chars)
            pred_entities = self.decode(sentence, pred_label_chars)

            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def convert_token_labels_to_char_labels(self, sentence, input_ids, token_labels):
        """
        token大小的标签转换为字符大小的标签
        """
        # 将input_ids转换为tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 去掉CLS和SEP
        if tokens[0] == "[CLS]":
            tokens = tokens[1:]
            token_labels = token_labels[1:]
        if tokens and tokens[-1] == "[SEP]":
            tokens = tokens[:-1]
            token_labels = token_labels[:-1]

        char_labels = [8] * len(sentence)  # 默认为O标签（值为8）

        char_pointer = 0

        for token_idx, token in enumerate(tokens):
            if token_idx >= len(token_labels):
                break
            if token_labels[token_idx] == -1:  # 忽略的标签
                continue

            if char_pointer >= len(sentence):
                break

            # 处理subword（以##开头的token）
            if token.startswith("##"):
                # 去掉##前缀
                token_text = token[2:]
                # 将标签分配给对应的字符
                for _ in range(len(token_text)):
                    if char_pointer < len(sentence):
                        char_labels[char_pointer] = token_labels[token_idx]
                        char_pointer += 1
                    else:
                        break
            else:
                # 正常token
                token_text = token

                # 在原文中查找匹配位置
                found = False
                search_start = char_pointer

                # 在一定范围内搜索匹配的位置
                for search_pos in range(search_start, min(search_start + 5, len(sentence))):
                    if self.match_token_at_position(sentence, search_pos, token_text):
                        # 找到匹配位置，将标签分配给对应的字符
                        for j in range(len(token_text)):
                            if search_pos + j < len(sentence):
                                char_labels[search_pos + j] = token_labels[token_idx]
                        char_pointer = search_pos + len(token_text)
                        found = True
                        break

                if not found:
                    # 如果没找到精确匹配，将标签分配给当前位置的字符
                    if char_pointer < len(sentence):
                        char_labels[char_pointer] = token_labels[token_idx]
                        char_pointer += 1

        return char_labels

    def match_token_at_position(self, text, start_pos, token_text):
        """
        检查在指定位置是否能匹配token文本
        """
        if start_pos + len(token_text) > len(text):
            return False

        # 直接字符串匹配
        return text[start_pos:start_pos + len(token_text)] == token_text

    def show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    '''
    {
      "B-LOCATION": 0,
      "B-ORGANIZATION": 1,
      "B-PERSON": 2,
      "B-TIME": 3,
      "I-LOCATION": 4,
      "I-ORGANIZATION": 5,
      "I-PERSON": 6,
      "I-TIME": 7,
      "O": 8
    }
    '''

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results