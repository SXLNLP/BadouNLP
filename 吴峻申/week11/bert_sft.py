import re
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig


class SFTLanguageModel(nn.Module):
    def __init__(self, input_dim, vocab, device='cpu'):
        super(SFTLanguageModel, self).__init__()
        # 确保vocab是字典类型
        assert isinstance(vocab, dict), "vocab必须是字典"
        self.device = device  # 添加device属性[1](@ref)

        # BERT配置
        config = BertConfig(
            vocab_size=len(vocab),
            hidden_size=input_dim,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=512,
            type_vocab_size=1
        )
        self.bert = BertModel(config)
        self.classify = nn.Linear(input_dim, len(vocab))
        self.loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.mask_cache = {}  # 掩码缓存
        self.to(device)  # 将模型移动到指定设备

    def create_sft_mask(self, seq_len, prompt_len):
        """向量化掩码计算（修复空数据集问题）"""
        cache_key = f"{seq_len}_{prompt_len}"
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        # 向量化操作替代循环
        mask = torch.ones(seq_len, seq_len, dtype=torch.float32)
        row_indices = torch.arange(seq_len).unsqueeze(1)
        col_indices = torch.arange(seq_len).unsqueeze(0)
        causal_mask = (row_indices >= col_indices) & (row_indices >= prompt_len)
        mask[prompt_len:, :prompt_len] = 1
        mask[prompt_len:, prompt_len:] = causal_mask[prompt_len:, prompt_len:].float()

        self.mask_cache[cache_key] = mask.unsqueeze(0)
        return self.mask_cache[cache_key]

    def forward(self, x, prompt_len=None, y=None):
        if y is not None and prompt_len is None:
            raise ValueError("训练模式下必须提供prompt_len")

        seq_len = x.shape[1]
        sft_mask = self.create_sft_mask(seq_len, prompt_len).to(self.device)

        outputs = self.bert(input_ids=x, attention_mask=sft_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classify(sequence_output)

        if y is not None:
            answer_start = prompt_len + 1
            # 边界检查[5](@ref)
            if answer_start >= seq_len:
                # 防止索引越界
                answer_start = max(0, seq_len - 2)

            answer_logits = logits[:, answer_start:-1, :]  # 忽略末尾[SEP]
            answer_labels = y[:, answer_start:-1]

            # 防止NaN损失[5](@ref)
            if torch.isnan(answer_logits).any():
                raise RuntimeError("发现NaN值在logits中")

            loss = self.loss(
                answer_logits.reshape(-1, logits.shape[-1]),
                answer_labels.reshape(-1)
            )
            return loss
        else:
            return torch.softmax(logits, dim=-1)


class SFTPairDataset(Dataset):
    def __init__(self, corpus, vocab, max_len=128):
        self.data = []
        self.vocab = vocab
        self.max_len = max_len
        self.prompt_lens = []

        # 确保语料有效
        if not corpus or len(corpus.strip()) == 0:
            print("警告：语料为空，使用示例数据")
            corpus = "李慕站在山路上，深深的呼吸"

        self.extract_qa_pairs(corpus)
        print(f"成功加载 {len(self.data)} 个问答对")

    def extract_qa_pairs(self, corpus):
        """健壮的问答对提取（修复分割失败问题）"""
        # 多种分句策略
        sentences = re.split(r'(?<=[。！？.!?])', corpus)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]  # 过滤短句

        # 确保有足够句子
        if len(sentences) < 2:
            # 回退策略：按行分割
            sentences = corpus.split('\n')
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

        # 生成问答对
        for i in range(0, len(sentences) - 1, 2):
            question = sentences[i]
            answer = sentences[i + 1] if i + 1 < len(sentences) else "暂无回答"
            self.add_qa_pair(question, answer)

    def add_qa_pair(self, question, answer):
        """添加问答对（带长度校验）"""
        # 跳过空问题或回答
        if len(question) == 0 or len(answer) == 0:
            return

        # 构建输入序列
        input_ids = [self.vocab["[CLS]"]]
        labels = [-100]

        # 问题部分
        for char in question:
            input_ids.append(self.vocab.get(char, self.vocab["<UNK>"]))
            labels.append(-100)
        input_ids.append(self.vocab["[SEP]"])
        labels.append(-100)
        prompt_len = len(input_ids) - 1  # 记录prompt长度

        # 回答部分
        for char in answer:
            input_ids.append(self.vocab.get(char, self.vocab["<UNK>"]))
            labels.append(self.vocab.get(char, self.vocab["<UNK>"]))
        input_ids.append(self.vocab["[SEP]"])
        labels.append(-100)

        # 截断或填充
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
            # 更新prompt_len确保在有效范围内
            prompt_len = min(prompt_len, self.max_len - 1)
        else:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.vocab["<pad>"]] * pad_len
            labels += [-100] * pad_len

        self.data.append((input_ids, labels))
        self.prompt_lens.append(prompt_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        prompt_len = self.prompt_lens[idx]
        return torch.LongTensor(x), torch.LongTensor(y), prompt_len


def build_vocab(vocab_path):
    """增强的词汇表构建（带自动生成）"""
    vocab = {"<pad>": 0, "<UNK>": 1, "<MASK>": 2, "[CLS]": 3, "[SEP]": 4}

    try:
        with open(vocab_path, encoding="utf8") as f:
            for line in f:
                char = line.strip()
                if char and char not in vocab:
                    vocab[char] = len(vocab)

        print(f"成功加载词汇表，大小: {len(vocab)}")
        return vocab
    except Exception as e:
        print(f"词汇表构建失败: {str(e)}，使用最小词汇表")
        return {"<pad>": 0, "<UNK>": 1, "<MASK>": 2, "[CLS]": 3, "[SEP]": 4}


def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def train_sft(save_weight=True):
    epoch_num = 20  # 减少轮数加速测试
    batch_size = 32
    char_dim = 256
    max_seq_len = 128

    vocab = build_vocab("vocab.txt")
    corpus = load_corpus("corpus.txt")
    print(f"语料长度: {len(corpus)} 字符")

    # 创建数据集
    dataset = SFTPairDataset(corpus, vocab, max_seq_len)
    if len(dataset) == 0:
        print("严重错误：数据集仍为空，请检查语料格式！")
        return

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用 {device} 训练")

    model = SFTLanguageModel(char_dim, vocab, device=device)  # 传入设备参数

    # 优化器（添加梯度裁剪）[8](@ref)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 创建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == 'cuda'
    )

    print("开始训练...")
    start_time = time.time()

    for epoch in range(epoch_num):
        model.train()
        total_loss = 0

        for batch_x, batch_y, batch_plens in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optim.zero_grad()
            loss = model(batch_x, prompt_len=batch_plens[0].item(), y=batch_y)

            # 检查NaN损失[5](@ref)
            if torch.isnan(loss):
                print("警告：发现NaN损失，跳过该批次")
                continue

            loss.backward()

            # 梯度裁剪防止梯度爆炸[8](@ref)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(loader))  # 避免除以零
        print(f"Epoch {epoch + 1}/{epoch_num} | Loss: {avg_loss:.4f}")
        print("生成示例:", generate_sft("深度学习的应用包括", model, vocab, max_seq_len))

    training_time = time.time() - start_time
    print(f"训练完成! 耗时: {training_time:.2f}秒")

    if save_weight:
        model_path = "bert_sft_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至: {model_path}")


def generate_sft(prompt, model, vocab, max_len=50):
    """优化的文本生成（修复device访问问题）"""
    model.eval()
    reverse_vocab = {v: k for k, v in vocab.items()}

    # 构建初始输入
    input_ids = [vocab["[CLS]"]]
    input_ids.extend([vocab.get(c, vocab["<UNK>"]) for c in prompt])
    input_ids.append(vocab["[SEP]"])
    prompt_len = len(input_ids) - 1

    # 预计算初始掩码（使用model.device）[1](@ref)
    seq_len = len(input_ids)
    sft_mask = model.create_sft_mask(seq_len, prompt_len).to(model.device)

    # 自回归生成
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.LongTensor([input_ids]).to(model.device)

            # 使用预计算掩码
            outputs = model.bert(x, attention_mask=sft_mask)
            logits = model.classify(outputs.last_hidden_state)
            next_token = torch.argmax(logits[0, -1]).item()

            if next_token == vocab["[SEP]"]:
                break
            input_ids.append(next_token)

            # 更新掩码
            if len(input_ids) > sft_mask.shape[1]:
                sft_mask = model.create_sft_mask(len(input_ids), prompt_len).to(model.device)

    # 解码答案
    answer_ids = input_ids[prompt_len + 1:]
    return ''.join([reverse_vocab.get(i, "<UNK>") for i in answer_ids])


if __name__ == "__main__":
    train_sft(save_weight=True)
