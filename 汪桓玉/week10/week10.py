import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import os
import random
import numpy as np
from tqdm import tqdm
import math


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, model_path):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path, return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, x, y=None):
        sequence_output = self.dropout(self.bert(input_ids=x)[0])
        y_pred = self.classify(sequence_output)
        
        if y is not None:
            return nn.functional.cross_entropy(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        return torch.softmax(y_pred, dim=-1)


def build_vocab(vocab_path):
    with open(vocab_path, encoding="utf8") as f:
        return {line.strip(): index for index, line in enumerate(f)}


def load_corpus(path):
    for encoding in ["utf-8", "gbk"]:
        try:
            with open(path, encoding=encoding) as f:
                return "".join(line.strip() for line in f)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode file {path}")

def apply_bert_mask(tokens, vocab, mask_prob=0.15):
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)

    for i in range(1, len(tokens) - 1):
        if random.random() < mask_prob:
            labels[i] = tokens[i]
            prob = random.random()
            if prob < 0.8:
                masked_tokens[i] = vocab["[MASK]"]
            elif prob < 0.9:
                masked_tokens[i] = random.randint(3, len(vocab) - 1)

    return masked_tokens, labels


def build_sample(vocab, window_size, corpus, tokenizer):
    start = random.randint(0, len(corpus) - window_size)
    window = corpus[start:start + window_size]
    encoded = tokenizer.encode(window, add_special_tokens=True)
    return apply_bert_mask(encoded, vocab)


def build_dataset_generator(sample_length, vocab, window_size, corpus, tokenizer):
    for _ in range(sample_length):
        yield build_sample(vocab, window_size, corpus, tokenizer)


def build_batch_dataset(batch_size, vocab, window_size, corpus, tokenizer):
    samples = list(build_dataset_generator(batch_size, vocab, window_size, corpus, tokenizer))
    dataset_x, dataset_y = zip(*samples)
    
    max_len = max(len(seq) for seq in dataset_x)
    padded_x = [x + [vocab["[PAD]"]] * (max_len - len(x)) for x in dataset_x]
    padded_y = [y + [-100] * (max_len - len(y)) for y in dataset_y]
    
    return torch.LongTensor(padded_x), torch.LongTensor(padded_y)


def build_model(vocab, model_path):
    return LanguageModel(len(vocab), model_path)


def generate_sentence(prefix, model, max_length=30):
    model.eval()
    device = next(model.parameters()).device
    
    input_ids = torch.LongTensor([model.tokenizer.encode(prefix, add_special_tokens=True)]).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]
            
            top_k_logits, top_k_indices = torch.topk(next_token_logits, 50)
            next_token_id = top_k_indices[torch.multinomial(torch.softmax(top_k_logits, dim=-1), 1)].item()
            
            if next_token_id == model.tokenizer.sep_token_id:
                break
                
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
    
    return model.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)


def calc_perplexity(sentence, model):
    model.eval()
    device = next(model.parameters()).device
    
    encoded = model.tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = torch.LongTensor([encoded]).to(device)
    
    with torch.no_grad():
        logits = model(input_ids)[0]
        probs = torch.softmax(logits, dim=-1)
        
        log_probs = [math.log(probs[i-1, encoded[i]].item()) for i in range(1, len(encoded))]
        
        return math.exp(-sum(log_probs) / len(log_probs)) if log_probs else float('inf')


def train(corpus_path, save_weight=True):
    epoch_num, batch_size, train_sample, window_size = 3, 16, 5000, 50
    model_path = "/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese"
    
    os.makedirs("model", exist_ok=True)
    
    vocab = build_vocab(f"{model_path}/vocab.txt")
    corpus = load_corpus(corpus_path)
    print(f"词汇表大小: {len(vocab)}, 语料长度: {len(corpus)}")
    
    model = build_model(vocab, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        progress_bar = tqdm(range(train_sample // batch_size), desc=f"Epoch {epoch + 1}/{epoch_num}")
        
        for _ in progress_bar:
            x, y = build_batch_dataset(batch_size, vocab, window_size, corpus, model.tokenizer)
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            watch_loss.append(loss.item())
            progress_bar.set_postfix({"loss": np.mean(watch_loss)})
        
        print(f"\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")

        for prefix in ["让他在半年之前，就不能做出", "李慕站在山路上，深深的呼吸"]:
            print(f"生成示例: {generate_sentence(prefix, model)}")
        
        ppl = calc_perplexity(corpus[:100], model)
        print(f"样本perplexity: {ppl:.4f}")
    
    if save_weight:
        save_path = os.path.join("model", os.path.basename(corpus_path).replace(".txt", ".pth"))
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存至 {save_path}")
    
    return model


if __name__ == "__main__":
    model_path = "/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese"
    corpus_path = "corpus.txt"
    
    if not os.path.exists(f"{model_path}/vocab.txt"):
        print(f"错误: vocab.txt文件不存在! 请检查路径: {model_path}")
    elif not os.path.exists(corpus_path):
        print(f"警告: 语料文件 {corpus_path} 不存在，请确保文件存在后再运行")
    else:
        train(corpus_path)