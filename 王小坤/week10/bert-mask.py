import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import numpy as np
import random
import os
import logging
from tqdm import tqdm

# 设置随机种子以确保结果可复现
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 自定义数据集类
class AutoregressiveMaskedLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        
        for text in texts:
            # 对文本进行编码
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.inputs.append(encodings)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        item = self.inputs[idx]
        input_ids = torch.tensor(item["input_ids"])
        attention_mask = torch.tensor(item["attention_mask"])
        
        # 创建标签，初始为-100（忽略计算损失）
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # 忽略padding token
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# 自回归BERT模型类
class AutoregressiveBertModel(nn.Module):
    def __init__(self, bert_model_name="bert-base-chinese", device="cuda"):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(bert_model_name)
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def autoregressive_generate(self, prompt, max_length=50, top_k=50, temperature=1.0):
        # 将提示文本转换为token IDs
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # 生成序列，直到达到最大长度或生成结束标记
        for _ in range(max_length):
            # 在序列末尾添加[MASK]标记
            mask_token_id = self.tokenizer.mask_token_id
            input_ids = torch.cat([input_ids, torch.tensor([[mask_token_id]]).to(self.device)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(self.device)], dim=1)
            
            # 获取模型对[MASK]标记的预测
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            # 获取最后一个[MASK]标记位置的预测
            next_token_logits = logits[0, -1, :]
            
            # 应用temperature
            next_token_logits = next_token_logits / temperature
            
            # 过滤top_k
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")
            
            # 计算概率分布
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            
            # 采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 将预测的token替换最后的[MASK]标记
            input_ids[0, -1] = next_token.item()
            
            # 如果生成了结束标记，则停止生成
            if next_token.item() == self.tokenizer.sep_token_id:
                break
        
        # 解码生成的序列
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text

# 训练函数
def train(model, train_dataloader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # 将数据移到设备上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

# 创建自回归掩码
def create_autoregressive_masks(input_ids, tokenizer):
    batch_size, seq_length = input_ids.size()
    device = input_ids.device
    
    # 创建自回归掩码数据
    ar_inputs = []
    ar_labels = []
    
    for i in range(batch_size):
        seq = input_ids[i].clone()
        
        # 找到序列中的非填充标记
        non_pad_indices = (seq != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            continue
            
        # 随机选择一个位置进行掩码（不包括CLS和SEP标记）
        valid_indices = non_pad_indices[1:-1]  # 排除CLS和SEP标记
        if len(valid_indices) == 0:
            continue
            
        # 随机选择掩码位置
        mask_pos = random.choice(valid_indices.tolist())
        
        # 创建输入序列（掩码之前的所有标记 + MASK标记）
        ar_input = seq.clone()
        ar_label = torch.ones_like(seq) * -100  # 初始化为-100（忽略）
        
        # 将mask_pos位置的标记设为MASK
        ar_input[mask_pos] = tokenizer.mask_token_id
        # 将mask_pos之后的所有标记也设为MASK
        for j in range(mask_pos + 1, seq_length):
            if seq[j] != tokenizer.pad_token_id and seq[j] != tokenizer.sep_token_id:
                ar_input[j] = tokenizer.mask_token_id
        
        # 设置标签（只预测mask_pos位置的标记）
        ar_label[mask_pos] = seq[mask_pos]
        
        ar_inputs.append(ar_input)
        ar_labels.append(ar_label)
    
    return torch.stack(ar_inputs), torch.stack(ar_labels)

# 自回归训练函数
def train_autoregressive(model, train_dataloader, optimizer, tokenizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # 将数据移到设备上
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # 创建自回归掩码
            ar_inputs, ar_labels = create_autoregressive_masks(input_ids, tokenizer)
            ar_inputs = ar_inputs.to(device)
            ar_labels = ar_labels.to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model.bert(input_ids=ar_inputs, attention_mask=attention_mask, labels=ar_labels)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载tokenizer和模型
    model_name = "bert-base-chinese"  # 可以根据需要更改为其他BERT模型
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = AutoregressiveBertModel(model_name, device).to(device)
    
    # 示例文本数据（实际应用中应该从文件加载）
    sample_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "深度学习是机器学习的分支，是一种以人工神经网络为架构，对数据进行表征学习的算法。",
        "自然语言处理是人工智能的一个子领域，专注于使计算机能够理解和生成人类语言。",
        "计算机视觉是人工智能的一个领域，研究如何使计算机能够从图像或视频中获取信息。",
        "强化学习是机器学习的一种方法，它通过让智能体在环境中采取行动并获得奖励或惩罚来学习最优策略。"
    ]
    
    # 创建数据集和数据加载器
    dataset = AutoregressiveMaskedLMDataset(sample_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 设置优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    
    # 训练模型
    logger.info("Starting autoregressive training...")
    train_autoregressive(model, dataloader, optimizer, tokenizer, device, epochs=3)
    
    # 保存模型
    output_dir = "./autoregressive_bert_model"
    os.makedirs(output_dir, exist_ok=True)
    model.bert.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    # 测试生成
    test_prompt = "人工智能是"
    logger.info(f"Generating text from prompt: '{test_prompt}'")
    generated_text = model.autoregressive_generate(test_prompt)
    logger.info(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()
