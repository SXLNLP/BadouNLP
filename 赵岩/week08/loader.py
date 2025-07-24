# loader.py
import json
import torch
import random
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TextMatchTripletDataset(Dataset):
    def __init__(self, data_path, config, shuffle=True):
        self.config = config
        self.max_length = config["max_length"]
        self.vocab = self.load_vocab(config["vocab_path"])
        self.data = self.load_data(data_path)
        self.knwb = self.build_knowledge_base()  # 构建知识库（标准问 <-> 变体）

    def build_knowledge_base(self):
        knwb = {}
        for idx, item in enumerate(self.data):
            questions = [self.encode_sentence(q) for q in item["questions"]]
            knwb[idx] = [torch.LongTensor(encoded) for encoded in questions]

        # 打印每组有多少个句子
        print("知识库各组样本数量：")
        for k, v in knwb.items():
            print(f"组 {k}: {len(v)} 个样本")

        return knwb

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = {line.strip(): idx for idx, line in enumerate(f)}
        return vocab

    @staticmethod
    def collate_fn(batch):
        """
        将多个 (anchor, positive, negative) 样本组合成 batch 形式
        """
        anchors, positives, negatives = zip(*batch)

        # 转换为 tensor 并 padding 成统一长度
        anchors = pad_sequence(anchors, batch_first=True).long()
        positives = pad_sequence(positives, batch_first=True).long()
        negatives = pad_sequence(negatives, batch_first=True).long()

        return anchors, positives, negatives

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if isinstance(item, dict) and "questions" in item and "target" in item:
                        data.append(item)
                    else:
                        print(f"警告: 跳过无效行 - {line}")
                except json.JSONDecodeError as e:
                    print(f"警告: 解析JSON失败 - {e}")
        return data

    def build_knowledge_base(self):
        """
        构建知识库：标准问 -> 所有变体句子的索引列表
        """
        knwb = {}
        self.schema = {}  # 标准问 -> index
        self.index_to_schema = {}  # index -> 标准问
        for idx, item in enumerate(self.data):
            target = item["target"]
            self.schema[target] = idx
            self.index_to_schema[idx] = target
            questions = [self.encode_sentence(q) for q in item["questions"]]
            knwb[idx] = [torch.LongTensor(encoded) for encoded in questions]
        return knwb

    def encode_sentence(self, text):
        input_id = []
        if self.config.get("use_jieba", False):  # 假设配置中有use_jieba选项来决定是否使用jieba分词
            from jieba import cut
        else:
            cut = lambda x: x
        for word_or_char in cut(text):
            input_id.append(self.vocab.get(word_or_char, self.vocab["[UNK]"]))
        input_id = input_id[:self.max_length]
        input_id += [0] * (self.max_length - len(input_id))  # padding
        return input_id

    def sample_triplet(self):
        # 随机选一个标准问组（正样本组）
        positive_group_idx = random.choice(list(self.knwb.keys()))
        positive_group = self.knwb[positive_group_idx]

        # 如果该组只有一个样本，无法选出不同的 anchor 和 positive，就重新采样整个组
        if len(positive_group) < 2:
            return self.sample_triplet()  # 递归调用直到找到合适的组

        # 正常采样
        anchor = random.choice(positive_group)
        positive = random.choice([q for q in positive_group if not torch.equal(q, anchor)])

        # 负样本组：与正样本组不同的组
        negative_group_idx = random.choice([x for x in self.knwb.keys() if x != positive_group_idx])
        negative = random.choice(self.knwb[negative_group_idx])

        return anchor, positive, negative

    def __len__(self):
        return self.config["epoch_data_size"]

    def __getitem__(self, item):
        return self.sample_triplet()


def load_data(data_path, config, shuffle=True):
    return TextMatchTripletDataset(data_path, config, shuffle=shuffle)
