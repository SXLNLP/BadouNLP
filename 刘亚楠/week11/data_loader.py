import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config import config


class TitleContentDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = self._load_data()

    def _load_data(self):
        with open(config.data_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        title = item['title']
        content = item['content']

        # 编码输入并截断
        title_ids = self.tokenizer.encode(title, add_special_tokens=False, max_length=config.max_length // 2,
                                          truncation=True)
        content_ids = self.tokenizer.encode(content, add_special_tokens=False, max_length=config.max_length // 2,
                                            truncation=True)

        # 构建输入序列
        input_ids = (
                [self.tokenizer.cls_token_id] +
                title_ids +
                [self.tokenizer.sep_token_id] +
                [self.tokenizer.convert_tokens_to_ids(config.bos_token)] +
                content_ids +
                [self.tokenizer.convert_tokens_to_ids(config.eos_token)]
        )

        # 确保不超过最大长度
        input_ids = input_ids[:config.max_length]
        original_len = len(input_ids)

        # 构建attention mask
        sep_pos = len(title_ids) + 1  # [CLS] + title + [SEP]
        attention_mask = torch.zeros((original_len, original_len))

        # 1. title部分双向注意力
        attention_mask[:sep_pos, :sep_pos] = 1

        # 2. content部分因果注意力
        for i in range(sep_pos, original_len):
            attention_mask[i, :i + 1] = 1

        # 构建loss mask
        loss_mask = [0] * (sep_pos + 1) + [1] * (original_len - sep_pos - 1)

        # 统一填充
        pad_len = config.max_length - original_len
        input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
        loss_mask = loss_mask + [0] * pad_len

        # 填充attention mask
        padded_attention_mask = torch.zeros((config.max_length, config.max_length))
        padded_attention_mask[:original_len, :original_len] = attention_mask

        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': padded_attention_mask,
            'loss_mask': torch.tensor(loss_mask)
        }



def print_data_sample(dataset, index=0):
    """打印数据样本的详细信息"""
    sample = dataset[index]
    tokenizer = dataset.tokenizer

    print("\n=== 数据样本检查 ===")
    print(f"原始数据: {dataset.data[index]}")

    # 解码input_ids
    decoded_input = tokenizer.decode(sample['input_ids'].tolist())
    print(f"\n完整输入 (长度: {len(sample['input_ids'])}):")
    print(decoded_input)

    # 可视化attention mask
    print("\nAttention Mask矩阵 (前20x20):")
    print(sample['attention_mask'][:20, :20].int().numpy())

    # 可视化loss mask
    print("\nLoss Mask (前20个位置):")
    print(sample['loss_mask'][:20].int().numpy())

    # 标识特殊token位置
    input_ids = sample['input_ids'].tolist()
    special_positions = {
        "CLS": input_ids.index(tokenizer.cls_token_id),
        "SEP": input_ids.index(tokenizer.sep_token_id),
        "BOS": input_ids.index(tokenizer.convert_tokens_to_ids(config.bos_token)),
        "EOS": input_ids.index(tokenizer.convert_tokens_to_ids(config.eos_token)) if tokenizer.convert_tokens_to_ids(
            config.eos_token) in input_ids else -1
    }
    print("\n特殊token位置:")
    for k, v in special_positions.items():
        print(f"{k}: {v} (token: {tokenizer.decode(input_ids[v]) if v != -1 else 'N/A'})")


if __name__ == "__main__":
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    tokenizer.add_special_tokens({'additional_special_tokens': [config.bos_token, config.eos_token]})

    # 创建数据集实例
    dataset = TitleContentDataset(tokenizer)

    # 打印数据集信息
    print(f"数据集大小: {len(dataset)}")
    print(f"示例数据 (索引0):")

    # 检查第一个样本
    print_data_sample(dataset, 0)

    # 检查随机样本
    import random

    random_index = random.randint(0, len(dataset) - 1)
    print("\n\n=== 随机样本检查 ===")
    print_data_sample(dataset, random_index)