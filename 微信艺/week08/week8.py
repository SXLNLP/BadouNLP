import os
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import jieba

# -------------------- 配置 --------------------
config = {
    "model_path": "model_output",
    "vocab_path": "chars.txt",  # 词汇表路径
    "train_data_path": "train.json",  # 训练数据路径
    "valid_data_path": "valid.json",  # 验证数据路径
    "schema_path": "schema.json",  # 标签映射路径
    "vocab_size": 1000,  # 根据实际词汇表更新
    "max_length": 20,  # 句子最大长度
    "hidden_size": 128,  # 模型隐藏层大小
    "epoch": 10,  # 训练轮次
    "batch_size": 32,  # 批大小
    "epoch_data_size": 1000,  # 每轮训练样本数
    "optimizer": "adam",  # 优化器
    "learning_rate": 1e-3,  # 学习率
}


# -------------------- 数据加载 --------------------
class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = self._load_vocab(config["vocab_path"])
        self.schema = self._load_schema(config["schema_path"])
        self.max_length = config["max_length"]
        self.data_type = None  # "train" or "test"
        self._load_data()

    def _load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0 for padding
        return token_dict

    def _load_schema(self, schema_path):
        with open(schema_path, encoding="utf8") as f:
            return json.loads(f.read())

    def _load_data(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):  # Train data
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self._encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                else:  # Test data
                    self.data_type = "test"
                    question, label = line
                    input_id = self._encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])

    def _encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self._padding(input_id)
        return input_id

    def _padding(self, input_id):
        input_id = input_id[:self.max_length]
        input_id += [0] * (self.max_length - len(input_id))
        return input_id

    def __len__(self):
        return self.config["epoch_data_size"] if self.data_type == "train" else len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self._random_triplet_sample()
        return self.data[index]

    def _random_triplet_sample(self):
        labels = list(self.knwb.keys())
        # Anchor and Positive
        anchor_label = random.choice(labels)
        anchor, positive = random.sample(self.knwb[anchor_label], 2)
        # Negative
        negative_label = random.choice([x for x in labels if x != anchor_label])
        negative = random.choice(self.knwb[negative_label])
        return anchor, positive, negative


def load_data(data_path, config, shuffle=True):
    dataset = DataGenerator(data_path, config)
    return DataLoader(dataset, batch_size=config["batch_size"], shuffle=shuffle)


# -------------------- 模型定义 --------------------
class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab_size"] + 1, config["hidden_size"], padding_idx=0)
        self.layer = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer(x)
        x = F.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        self.device = torch.device("cpu")  # 将在训练时被覆盖

    def forward(self, anchor, positive=None, negative=None):
        if positive is not None and negative is not None:
            anchor_vec = self.sentence_encoder(anchor)
            positive_vec = self.sentence_encoder(positive)
            negative_vec = self.sentence_encoder(negative)
            return self.triplet_loss(anchor_vec, positive_vec, negative_vec)
        return self.sentence_encoder(anchor)  # 推理时仅编码句子


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    lr = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr)


# -------------------- 训练与评估 --------------------
class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}

    def _build_knowledge_vectors(self):
        self.question_index_to_label = {}
        self.question_ids = []
        for label, questions in self.train_data.dataset.knwb.items():
            for q in questions:
                self.question_index_to_label[len(self.question_ids)] = label
                self.question_ids.append(q)
        with torch.no_grad():
            question_matrix = torch.stack(self.question_ids, dim=0).to(self.model.device)
            self.knwb_vectors = self.model(question_matrix)
            self.knwb_vectors = F.normalize(self.knwb_vectors, dim=-1)

    def evaluate(self, epoch):
        self.model.eval()
        self._build_knowledge_vectors()
        self.stats_dict = {"correct": 0, "wrong": 0}

        for batch in self.valid_data:
            input_ids, labels = batch
            input_ids = input_ids.to(self.model.device)
            with torch.no_grad():
                test_vecs = self.model(input_ids)
                sims = torch.mm(test_vecs, self.knwb_vectors.T)
                preds = torch.argmax(sims, dim=1)
                for pred, label in zip(preds, labels):
                    if self.question_index_to_label[pred.item()] == label.item():
                        self.stats_dict["correct"] += 1
                    else:
                        self.stats_dict["wrong"] += 1

        acc = self.stats_dict["correct"] / (self.stats_dict["correct"] + self.stats_dict["wrong"])
        self.logger.info(f"Epoch {epoch}: Accuracy = {acc:.4f}")


# -------------------- 主程序 --------------------
def main():
    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 创建模型目录
    os.makedirs(config["model_path"], exist_ok=True)

    # 加载数据
    train_data = load_data(config["train_data_path"], config)

    # 初始化模型
    model = SiameseNetwork(config)

    # 设备设置
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    model.device = device  # 供评估器使用
    logger.info(f"Using device: {device}")

    # 优化器
    optimizer = choose_optimizer(config, model)

    # 评估器
    evaluator = Evaluator(config, model, logger)

    # 训练循环
    for epoch in range(1, config["epoch"] + 1):
        model.train()
        total_loss = 0

        for anchor, pos, neg in train_data:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            loss = model(anchor, pos, neg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # 每2轮评估一次
        if epoch % 2 == 0:
            evaluator.evaluate(epoch)
            torch.save(model.state_dict(), f"{config['model_path']}/epoch_{epoch}.pth")


if __name__ == "__main__":
    main()