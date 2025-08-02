# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)

    # 加载模型
    model = TorchModel(config)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("使用设备: {}".format(device))

    # 加载优化器
    optimizer = choose_optimizer(config, model)

    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)

    # 训练参数
    num_epochs = config["epoch"]
    best_loss = float('inf')

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = []
        logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))

        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()

            # 移动数据到设备
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch_data['labels'].to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0] if isinstance(outputs, tuple) else outputs

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if index % max(1, int(len(train_data) / 2)) == 0:
                logger.info("batch {} loss: {:.4f}".format(index, loss.item()))

        # 计算并记录epoch平均损失
        epoch_loss = np.mean(train_loss)
        logger.info("epoch {} average loss: {:.4f}".format(epoch + 1, epoch_loss))

        # 评估模型
        evaluator.eval(epoch)

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model_path = os.path.join(config["model_path"], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info("保存最佳模型到: {}".format(model_path))

    # 保存最终模型
    final_model_path = os.path.join(config["model_path"], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info("保存最终模型到: {}".format(final_model_path))

    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)
  --------------
# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)


    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset.sentences[index * self.config["batch_size"]: (index+1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, attention_mask, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_id,attention_mask)
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

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
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
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


------
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert_path = config["bert_path"]
        self.hidden_size = config["hidden_size"]
        self.class_num = config["class_num"]

        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.class_num)
        self.use_crf = config["use_crf"]
        if self.use_crf:
            self.crf_layer = CRF(self.class_num, batch_first=True)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state      #(batch_size, sen_len, hidden_size)
        predict = self.classify(sequence_output) #(batch_size, seq_len, class_num)

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1)
                return - self.crf_layer(predict, labels, mask, reduction="mean")
            else:
                predict = predict.view(-1, predict.shape[-1])  # (batch_size*seq_len, class_num)
                target = labels.view(-1)# (batch_size*seq_len)
                return self.loss(predict, target)
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
  ----
# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""

from transformers import BertTokenizer
import torch


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])  # 使用BERT的tokenizer
        self.schema = self.load_schema(config["schema_path"])
        self.data = []
        self.load()

    def load_schema(self, schema_path):
        with open(schema_path, 'r', encoding='utf8') as f:
            schema = json.load(f)
        if not isinstance(schema, dict):
            raise ValueError("Schema should be a dictionary mapping labels to indices")

        return schema

    def load(self):
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                if segment.strip() == "":
                    continue
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        char, label = parts[0], parts[1]
                        sentence.append(char)
                        labels.append(self.schema[label])

                if sentence:  # 确保句子不为空
                    input_ids, attention_mask = self.encode_sentence("".join(sentence))
                    labels = self.padding(labels, -1)
                    self.data.append({
                        'input_ids': torch.LongTensor(input_ids),
                        'attention_mask': torch.LongTensor(attention_mask),
                        'labels': torch.LongTensor(labels)
                    })

    def encode_sentence(self, text):
        # 使用BERT的tokenizer进行编码
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0).tolist(), encoded['attention_mask'].squeeze(0).tolist()

    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        input_ids = []
        attention_masks = []
        labels = []

        for data in batch:
            input_ids.append(data['input_ids'])
            attention_masks.append(data['attention_mask'])
            labels.append(data['labels'])

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }


# 用torch自带的DataLoader类封装数据


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle, collate_fn=dg.collate_fn)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("../ner_data/train.txt", Config)
  ----
# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 128,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"E:\PycharmProjects\NLPtask\week9序列标注问题\Practice\bert-base-chinese"
}

