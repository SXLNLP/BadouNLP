# coding:utf8
import json
import os
import random

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class SFTLanguageModel(nn.Module):
    """
    基于BERT的SFT语言模型，用于半自动标题-正文生成任务
    """

    def __init__(self, hidden_size, vocab_size, pretrain_model_path, sep_token_id=102):
        super(SFTLanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.functional.cross_entropy
        self.sep_token_id = sep_token_id

    def create_sft_mask(self, input_ids):
        """
        生成attention mask：
        前段（标题）完全可见；
        后段（正文）只能看到自己和标题，模拟自回归和知识注入。
        """
        batch_size, seq_len = input_ids.shape
        mask = torch.zeros(batch_size, seq_len, seq_len, device=input_ids.device)

        sep_positions = (input_ids == self.sep_token_id).nonzero(as_tuple=False)
        first_sep_pos = {}

        for pos in sep_positions:
            sample_idx, token_pos = pos[0].item(), pos[1].item()
            if sample_idx not in first_sep_pos:
                first_sep_pos[sample_idx] = token_pos

        for i in range(batch_size):
            if i in first_sep_pos:
                sep = min(first_sep_pos[i], seq_len - 2)
                mask[i, :sep + 1, :sep + 1] = 1
                mask[i, sep + 1:, :sep + 1] = 1
                content_len = seq_len - sep - 1
                if content_len > 0:
                    tril = torch.tril(torch.ones(content_len, content_len, device=input_ids.device))
                    mask[i, sep + 1:, sep + 1:] = tril
            else:
                mask[i] = torch.ones(seq_len, seq_len, device=input_ids.device)

        return mask

    def forward(self, input_ids, labels=None):
        """
        如果labels存在，则计算loss；
        否则返回预测结果（使用 generate）。
        """
        if labels is not None:
            mask = self.create_sft_mask(input_ids)
            outputs, _ = self.bert(input_ids, attention_mask=mask)
            logits = self.classify(outputs)

            total_loss, valid_samples = 0, 0
            sep_mask = (input_ids == self.sep_token_id)

            for i in range(input_ids.size(0)):
                if sep_mask[i].any():
                    sep = sep_mask[i].nonzero()[0].item()
                    content_logits = logits[i, sep + 1:]
                    content_labels = labels[i, sep + 1:]
                    if content_logits.size(0) == content_labels.size(0):
                        loss = self.loss_fn(
                            content_logits.view(-1, content_logits.size(-1)),
                            content_labels.view(-1)
                        )
                        total_loss += loss
                        valid_samples += 1
                else:
                    loss = self.loss_fn(
                        logits[i].view(-1, logits.size(-1)),
                        labels[i].view(-1)
                    )
                    total_loss += loss
                    valid_samples += 1

            if torch.isnan(total_loss):
                return torch.tensor(0.0, device=input_ids.device)
            return total_loss / valid_samples if valid_samples > 0 else torch.tensor(0.0)
        else:
            return self.generate(input_ids)

    def generate(self, input_ids, max_length=512, temperature=1.0, tokenizer=None):
        """
        基于标题 + [SEP] 生成正文内容。
        """
        sep_pos = (input_ids[0] == self.sep_token_id).nonzero()
        sep_pos = sep_pos[0, 0].item() if sep_pos.numel() > 0 else input_ids.size(1) - 1
        generated = []

        for _ in range(max_length):
            mask = self.create_sft_mask(input_ids)
            outputs, _ = self.bert(input_ids, attention_mask=mask)
            logits = self.classify(outputs[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, 50)
            next_token = top_indices[0, torch.multinomial(top_probs, 1)]

            if next_token.item() == self.sep_token_id:
                break

            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=-1)
            generated.append(next_token.item())

        if tokenizer and generated:
            return tokenizer.decode(generated, skip_special_tokens=True).strip()
        return "无法生成内容"


def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path, 'r', encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            corpus.append((item["title"], item["content"]))
    return corpus


def build_sft_sample(tokenizer, title, content, max_length=128):
    if len(content) < 10:
        return None, None

    text = title + "[SEP]" + content
    inputs = tokenizer(text, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
    labels = inputs['input_ids'].clone()

    sep_pos = (inputs['input_ids'][0] == tokenizer.sep_token_id).nonzero()
    if sep_pos.numel() == 0:
        return None, None

    sep = sep_pos[0].item()
    labels[:, :sep + 1] = -100
    if inputs['input_ids'].size(1) - sep - 1 < 10:
        return None, None

    return inputs['input_ids'], labels


def build_model(hidden_size, vocab_size, pretrain_model_path):
    return SFTLanguageModel(hidden_size, vocab_size, pretrain_model_path)


def generate_sentence(model, tokenizer, title, max_length=512):
    input_text = title + "[SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').cuda() if torch.cuda.is_available() else None
    return model.generate(input_ids, max_length, tokenizer=tokenizer)


def train(corpus_path: str, pretrain_model_path: str, save_weight=False):
    epoch_num = 5
    batch_size = 64
    train_sample = 10000
    window_size = 128
    char_dim = 768
    vocab_size = 21128
    learning_rate = 1e-5

    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)
    corpus = load_corpus(corpus_path)
    model = build_model(char_dim, vocab_size, pretrain_model_path)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("加载完成，开始训练...\n")

    for epoch in range(epoch_num):
        model.train()
        random.shuffle(corpus)
        total_loss, batch_count = 0, 0

        for i in range(0, min(len(corpus), train_sample), batch_size):
            batch = corpus[i:i + batch_size]
            x_list, y_list = [], []

            for title, content in batch:
                x, y = build_sft_sample(tokenizer, title, content, max_length=window_size)
                if x is not None and y is not None:
                    x_list.append(x)
                    y_list.append(y)

            if not x_list:
                continue

            input_ids = torch.stack(x_list).squeeze(1)
            labels = torch.stack(y_list).squeeze(1)

            if torch.cuda.is_available():
                input_ids, labels = input_ids.cuda(), labels.cuda()

            optimizer.zero_grad()
            loss = model(input_ids, labels)

            if not torch.isnan(loss) and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1

            if batch_count % 50 == 0:
                avg = total_loss / max(1, batch_count)
                print(f"[Epoch {epoch + 1}] batch {batch_count} average loss: {avg:.4f}")

        print(f"Epoch {epoch + 1} 结束，平均损失: {total_loss / max(batch_count, 1):.4f}")

    if save_weight:
        os.makedirs("model", exist_ok=True)
        model_path = os.path.join("model", os.path.basename(corpus_path).replace(".json", ".pth"))
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train(corpus_path="./corpus.json", pretrain_model_path="bert-base-chinese")