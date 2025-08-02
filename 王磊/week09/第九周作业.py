# -*- coding: utf-8 -*-
import os
import json
import re
import logging
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from transformers import BertModel, BertTokenizer

# 配置参数信息
class Config:
    def __init__(self):
        self.model_path = "model_output"
        self.schema_path = "ner_data/schema.json"
        self.train_data_path = "ner_data/train.txt"
        self.valid_data_path = "ner_data/test.txt"
        self.vocab_path = "chars.txt"
        self.max_length = 128
        self.hidden_size = 768
        self.num_layers = 1
        self.epoch = 10
        self.batch_size = 32
        self.optimizer = "adam"
        self.learning_rate = 5e-5
        self.use_crf = True
        self.class_num = 9
        self.use_bert = True
        self.bert_path = r"E:\ailearn\bert\bert-base-chinese"

# 数据加载器
class NERDataLoader:
    def __init__(self, config):
        self.config = config
        if self.config.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
            self.vocab = self.tokenizer.get_vocab()
        else:
            self.vocab = self.load_vocab(config.vocab_path)
        self.schema = self.load_schema(config.schema_path)
        self.sentences = []

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        return token_dict

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

    def load_data(self, data_path, shuffle=True):
        data = []
        with open(data_path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(sentence))
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, -1)
                data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])

        return DataLoader(data, batch_size=self.config.batch_size, shuffle=shuffle)

    def encode_sentence(self, text):
        if self.config.use_bert:
            # 对每个字符进行tokenize，注意处理中文
            text = "".join(text)  # 中文BERT是按字处理的
            input_id = self.tokenizer.encode(text,
                                          add_special_tokens=True,
                                          max_length=self.config.max_length,
                                          truncation=True)
            # 对于NER任务，我们不需要[CLS]和[SEP]的标签
            input_id = self.padding(input_id)
            return input_id
        else:
            input_id = []
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
            return self.padding(input_id)

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config.max_length]
        input_id += [pad_token] * (self.config.max_length - len(input_id))
        return input_id

# 模型结构
class BERTNERModel(nn.Module):
    def __init__(self, config):
        super(BERTNERModel, self).__init__()
        self.config = config

        if config.use_bert:
            self.bert = BertModel.from_pretrained(config.bert_path)
            hidden_size = self.bert.config.hidden_size
        else:
            hidden_size = config.hidden_size
            vocab_size = len(self.load_vocab(config.vocab_path)) + 1
            self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
            self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True,
                               bidirectional=True, num_layers=config.num_layers)
            hidden_size *= 2  # 双向LSTM

        self.classify = nn.Linear(hidden_size, config.class_num)

        if config.use_crf:
            self.crf_layer = CRF(config.class_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1
        return token_dict

    def forward(self, x, target=None):
        if self.config.use_bert:
            attention_mask = (x != 0).float()  # 创建attention mask
            outputs = self.bert(x, attention_mask=attention_mask)
            x = outputs[0]  # 取最后一层的输出 [batch_size, seq_len, hidden_size]
        else:
            x = self.embedding(x)
            x, _ = self.layer(x)

        logits = self.classify(x)  # [batch_size, seq_len, class_num]

        if target is not None:
            if self.config.use_crf:
                mask = (target != -1)  # 创建mask，忽略padding部分
                return -self.crf_layer(logits, target, mask, reduction="mean")
            else:
                active_loss = attention_mask.view(-1) == 1 if self.config.use_bert else (target.view(-1) != -1)
                active_logits = logits.view(-1, self.config.class_num)[active_loss]
                active_labels = target.view(-1)[active_loss]
                return self.loss(active_logits, active_labels)
        else:
            if self.config.use_crf:
                return self.crf_layer.decode(logits)
            else:
                return torch.argmax(logits, dim=-1)

# 评估器
class NEREvaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.data_loader = NERDataLoader(config)
        self.valid_data = self.data_loader.load_data(config.valid_data_path, shuffle=False)
        self.schema = {v:k for k,v in self.data_loader.load_schema(config.schema_path).items()}
        self.stats_dict = {"LOCATION": defaultdict(int),
                          "TIME": defaultdict(int),
                          "PERSON": defaultdict(int),
                          "ORGANIZATION": defaultdict(int)}

    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果：")
        self._reset_stats()
        self.model.eval()

        for index, batch_data in enumerate(self.valid_data):
            sentences = self.data_loader.sentences[index * self.config.batch_size: (index+1) * self.config.batch_size]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data

            with torch.no_grad():
                pred_results = self.model(input_id)

            self._write_stats(labels, pred_results, sentences)

        self._show_stats()

    def _reset_stats(self):
        for key in self.stats_dict:
            for sub_key in ["正确识别", "样本实体数", "识别出实体数"]:
                self.stats_dict[key][sub_key] = 0

    def _write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)

        if not self.config.use_crf:
            pred_results = torch.argmax(pred_results, dim=-1)

        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config.use_crf:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()

            # 处理BERT的特殊token
            if self.config.use_bert:
                true_label = true_label[1:1+len(sentence)]  # 去掉[CLS]和[SEP]的标签
                pred_label = pred_label[1:1+len(sentence)]
            else:
                true_label = true_label[:len(sentence)]
                pred_label = pred_label[:len(sentence)]

            true_entities = self._decode(sentence, true_label)
            pred_entities = self._decode(sentence, pred_label)

            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])

    def _decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)

        # 解码规则需要根据实际的标签映射调整
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

    def _show_stats(self):
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info(f"{key}类实体，准确率：{precision:.4f}, 召回率: {recall:.4f}, F1: {F1:.4f}")

        macro_f1 = np.mean(F1_scores)
        self.logger.info(f"Macro-F1: {macro_f1:.4f}")

        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])

        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)

        self.logger.info(f"Micro-F1: {micro_f1:.4f}")
        self.logger.info("--------------------")

# 训练主程序
class NERTrainer:
    def __init__(self, config):
        self.config = config
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self._setup()

    def _setup(self):
        # 设置随机种子
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True

        # 创建模型目录
        if not os.path.isdir(self.config.model_path):
            os.makedirs(self.config.model_path)

    def train(self):
        # 初始化数据加载器
        data_loader = NERDataLoader(self.config)
        train_data = data_loader.load_data(self.config.train_data_path)

        # 初始化模型
        model = BERTNERModel(self.config)
        if torch.cuda.is_available():
            model = model.cuda()
            self.logger.info("使用GPU训练")

        # 初始化优化器
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate) if self.config.optimizer == "adam" else SGD(model.parameters(), lr=self.config.learning_rate)

        # 初始化评估器
        evaluator = NEREvaluator(self.config, model, self.logger)

        # 训练循环
        best_f1 = 0
        for epoch in range(self.config.epoch):
            epoch += 1
            model.train()
            self.logger.info(f"epoch {epoch} begin")
            train_loss = []

            for index, batch_data in enumerate(train_data):
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    batch_data = [d.cuda() for d in batch_data]

                input_id, labels = batch_data
                loss = model(input_id, labels)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                if index % int(len(train_data) / 2) == 0:
                    self.logger.info(f"batch loss {loss.item():.4f}")

            avg_loss = np.mean(train_loss)
            self.logger.info(f"epoch average loss: {avg_loss:.4f}")

            # 评估
            evaluator.eval(epoch)

            # 保存最好的模型
            current_f1 = np.mean([evaluator.stats_dict[key]["正确识别"] / (1e-5 + evaluator.stats_dict[key]["样本实体数"])
                                for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
            if current_f1 > best_f1:
                best_f1 = current_f1
                model_path = os.path.join(self.config.model_path, "best_model.pth")
                torch.save(model.state_dict(), model_path)
                self.logger.info("保存当前最优模型")

# 主函数
if __name__ == "__main__":
    config = Config()
    trainer = NERTrainer(config)
    trainer.train()
