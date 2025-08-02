# coding:utf8
"""
基于 BERT + causal mask 的自回归语言模型（无 numpy 依赖）
"""
import torch
import torch.nn as nn
import random
import math
import os
from transformers import BertConfig, BertModel

# ---------------------------- 模型 ----------------------------
class BertLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=4, max_len=512):
        super(BertLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_len, hidden_size)

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_len,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.encoder = BertModel(config).encoder
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, y=None):
        bsz, seq_len = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)

        # causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        emb = self.embedding(x) + self.position_embedding(pos)
        hidden = self.encoder(emb, attention_mask=mask).last_hidden_state
        logits = self.classifier(hidden)

        if y is not None:
            return self.loss(logits.view(-1, logits.size(-1)), y.view(-1))
        return torch.softmax(logits, dim=-1)

# ---------------------------- 数据 ----------------------------
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for idx, line in enumerate(f):
            char = line.strip()
            vocab[char] = idx + 1
    return vocab

def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - window_size - 1)
    window = corpus[start:start + window_size]
    target = corpus[start + 1:start + window_size + 1]
    x = [vocab.get(c, 0) for c in window]
    y = [vocab.get(c, 0) for c in target]
    return x, y

def build_dataset(sample_len, vocab, window_size, corpus):
    xs, ys = [], []
    for _ in range(sample_len):
        x, y = build_sample(vocab, window_size, corpus)
        xs.append(x)
        ys.append(y)
    return torch.LongTensor(xs), torch.LongTensor(ys)

# ---------------------------- 推理 ----------------------------
def generate_sentence(start, model, vocab, window_size):
    rev = {v: k for k, v in vocab.items()}
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        out = start
        while len(out) <= 30:
            ids = [vocab.get(c, 0) for c in out[-window_size:]]
            x = torch.LongTensor([ids]).to(device)
            prob = model(x)[0, -1]
            idx = sampling_strategy(prob)
            nxt = rev[idx]
            if nxt == "\n":
                break
            out += nxt
        return out

def sampling_strategy(prob):
    if random.random() > 0.1:
        return int(torch.argmax(prob))
    return int(torch.multinomial(prob, 1))

# ---------------------------- 训练 ----------------------------
def train(corpus_path, save_weight=True):
    epoch_num = 20
    batch_size = 64
    train_sample = 50000
    char_dim = 256
    window_size = 10
    vocab = build_vocab("vocab.txt")
    corpus = load_corpus(corpus_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertLM(len(vocab), hidden_size=char_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        losses = []
        for _ in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f"=========\n第{epoch + 1}轮平均loss:{sum(losses)/len(losses):.4f}")
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))

    if save_weight:
        os.makedirs("model", exist_ok=True)
        path = os.path.join("model", os.path.basename(corpus_path).replace("txt", "pth"))
        torch.save(model.state_dict(), path)

# ---------------------------- 入口 ----------------------------
if __name__ == "__main__":
    train("corpus.txt", save_weight=True)
