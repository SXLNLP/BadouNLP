#week11作业
#使用新闻数据尝试实现sft训练。

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)

# 1. 加载新闻数据集
def load_news_dataset(file_path, sample_size=1000):
    """加载新闻数据集并格式化为指令-响应对"""
    try:
        # 尝试读取不同格式的新闻数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path).sample(sample_size)
            # 假设CSV包含'title'和'content'列
            return [{"instruction": f"写一篇关于'{row['title']}'的新闻", "response": row['content']} 
                    for _, row in df.iterrows()]
        
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return [{"instruction": f"总结这篇新闻: {item['headline']}", "response": item['article']} 
                    for item in data[:sample_size]]
        
        else:
            raise ValueError("Unsupported file format")
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # 创建示例数据集作为备用
        return [
            {"instruction": "写一篇关于人工智能最新进展的新闻", "response": "近日，研究人员宣布在自然语言处理领域取得重大突破..."},
            {"instruction": "报道一下最新的科技新闻", "response": "科技巨头公司今日发布了最新一代智能手机..."},
            {"instruction": "写一篇关于气候变化的新闻报道", "response": "联合国最新报告显示，全球气温上升速度超出预期..."},
            {"instruction": "总结一下最新的体育新闻", "response": "在昨晚举行的冠军赛中，卫冕冠军意外失利..."},
            {"instruction": "报道一条国际政治新闻", "response": "多国领导人今日就全球贸易协定达成共识..."}
        ] * 200  # 扩展数据集

# 2. 创建自定义数据集
class NewsSFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"[INST] {item['instruction']} [/INST]"
        response = item['response']
        
        # 拼接提示和响应
        full_text = f"{prompt} {response}"
        
        # 编码文本
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建标签 - 仅对响应部分计算损失
        input_ids = encoding["input_ids"].squeeze()
        labels = input_ids.clone()
        
        # 计算提示部分的长度（不包括响应）
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]
        
        # 将提示部分设置为-100（损失忽略）
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }

# 3. 创建SFT模型包装器
class SFTModel(nn.Module):
    def __init__(self, base_model_name="microsoft/phi-2", freeze_layers=0):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # 冻结部分层（可选）
        if freeze_layers > 0:
            num_layers = len(self.model.model.layers)
            for i, layer in enumerate(self.model.model.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            print(f"Froze first {freeze_layers} layers out of {num_layers}")
        
        # 启用梯度检查点（节省显存）
        self.model.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        ).loss

# 4. 训练函数
def train_sft(model, train_loader, val_loader, epochs=5, lr=2e-5):
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # 训练指标
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        train_loss = 0
        
        # 训练阶段
        train_progress = tqdm(train_loader, desc="Training")
        for batch in train_progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 前向传播
            loss = model(input_ids, attention_mask, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            # 记录损失
            train_loss += loss.item()
            train_progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                loss = model(input_ids, attention_mask, labels)
                val_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_sft_model.pth")
            print("Saved best model")
    
    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SFT Training Progress")
    plt.legend()
    plt.savefig("sft_training_plot.png")
    plt.show()
    
    return model

# 5. 文本生成函数
def generate_news(model, tokenizer, instruction, max_length=200, temperature=0.7):
    model.eval()
    prompt = f"[INST] {instruction} [/INST]"
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 生成配置
    generation_config = {
        "max_new_tokens": max_length,
        "temperature": temperature,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }
    
    # 生成文本
    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=input_ids,
            **generation_config
        )
    
    # 解码并提取响应部分
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("[/INST]")[-1].strip()
    
    return response

# 主函数
def main():
    # 加载数据集
    dataset = load_news_dataset("news_data.json")  # 替换为您的新闻数据集
    print(f"Loaded {len(dataset)} news examples")
    
    # 初始化分词器和模型
    model_name = "microsoft/phi-2"  # 轻量级但性能强大的模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置填充令牌
    
    # 创建数据集
    sft_dataset = NewsSFTDataset(dataset, tokenizer)
    
    # 划分训练集和验证集
    train_size = int(0.9 * len(sft_dataset))
    val_size = len(sft_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        sft_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = SFTModel(model_name, freeze_layers=10).to(device)
    
    # 启用PyTorch 2.x的编译加速（可选）
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    except Exception as e:
        print(f"Compilation warning: {e}")
    
    # 打印可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 训练模型
    trained_model = train_sft(
        model, 
        train_loader, 
        val_loader, 
        epochs=5,
        lr=5e-6
    )
    
    # 保存最终模型
    torch.save(trained_model.state_dict(), "sft_news_model.pth")
    print("Saved final model")
    
    # 测试生成
    test_instructions = [
        "写一篇关于量子计算突破的新闻",
        "报道一下最新的体育赛事结果",
        "写一篇关于环保科技创新的新闻",
        "总结一下最新的国际政治动态"
    ]
    
    print("\nTest Generations:")
    for instruction in test_instructions:
        print(f"\nInstruction: {instruction}")
        response = generate_news(trained_model, tokenizer, instruction)
        print(f"Response: {response}\n")
        print("-" * 80)

if __name__ == "__main__":
    main()
