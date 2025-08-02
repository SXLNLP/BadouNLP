# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer  # 引入Bert分词器

class DataGenerator(Dataset):  # 继承Dataset更规范
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.vocab = self.tokenizer.get_vocab()
        # 强制更新config的vocab_size为Bert词汇表大小
        config["vocab_size"] = len(self.vocab)  # 关键：确保这里正确赋值
        logger.info(f"Bert词汇表大小：{config['vocab_size']}")  # 打印确认（应显示21128左右）
        self.config["pad_idx"] = self.tokenizer.pad_token_id
        self.config["start_idx"] = self.tokenizer.cls_token_id
        self.config["end_idx"] = self.tokenizer.sep_token_id
        self.data = self.load()

    def load(self):
        data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                # 处理输入（新闻内容）和输出（标题）
                input_dict = self.tokenizer(
                    content,
                    max_length=self.config["input_max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                # 标题作为目标，需添加起始符[CLS]和结束符[SEP]
                target_dict = self.tokenizer(
                    title,
                    max_length=self.config["output_max_length"]-2,  # 预留起始和结束符
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                # 目标序列：[CLS] + 标题 + [SEP]（自回归输入）
                target_ids = torch.cat([
                    torch.tensor([self.config["start_idx"]]),  # 起始符
                    target_dict["input_ids"][0],
                    torch.tensor([self.config["end_idx"]])     # 结束符
                ], dim=0)
                # 真实标签（用于计算loss，与目标序列错开一位）
                gold_ids = torch.cat([
                    target_dict["input_ids"][0],
                    torch.tensor([self.config["end_idx"], self.config["pad_idx"]])  # 补位
                ], dim=0)[:self.config["output_max_length"]]  # 截断到最大长度
                # 组装数据（input_ids, attention_mask, target_ids, gold_ids）
                data.append({
                    "input_ids": input_dict["input_ids"][0],
                    "attention_mask": input_dict["attention_mask"][0],
                    "target_ids": target_ids,
                    "gold_ids": gold_ids
                })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

# 封装为DataLoader
def load_data(data_path, config, logger, shuffle=True):
    dataset = DataGenerator(data_path, config, logger)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=lambda x: {  # 批量处理字典格式数据
            "input_ids": torch.stack([d["input_ids"] for d in x]),
            "attention_mask": torch.stack([d["attention_mask"] for d in x]),
            "target_ids": torch.stack([d["target_ids"] for d in x]),
            "gold_ids": torch.stack([d["gold_ids"] for d in x])
        }
    )
    return dataloader

if __name__ == "__main__":
    from config import Config
    import logging
    logger = logging.getLogger()
    dl = load_data(Config["train_data_path"], Config, logger)
    sample = next(iter(dl))
    print("输入ID形状：", sample["input_ids"].shape)  # (batch, input_max_length)
    print("目标ID形状：", sample["target_ids"].shape)  # (batch, output_max_length)