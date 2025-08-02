import os
import json
import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


# 设置随机种子确保可复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# 定义标题到内容生成模型
class Title2ContentModel(nn.Module):
    def __init__(self, pretrain_model_path):
        super(Title2ContentModel, self).__init__()
        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(
            pretrain_model_path,
            return_dict=False,
            attn_implementation='eager'
        )
        # 输出层：预测下一个token
        self.output_layer = nn.Linear(
            self.bert.config.hidden_size,
            self.bert.config.vocab_size
        )
        # 损失函数（忽略标签为0的位置）
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        # 冻结BERT部分层（加速训练，防止过拟合）
        for param in list(self.bert.parameters())[:100]:
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        # BERT编码
        bert_output, _ = self.bert(input_ids, attention_mask=attention_mask)
        # 预测token
        logits = self.output_layer(bert_output)  # (batch_size, seq_len, vocab_size)

        if labels is not None:
            # 计算损失
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss
        else:
            # 预测时返回概率分布
            return torch.softmax(logits, dim=-1)


# 自定义数据集
class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = tokenizer.sep_token
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        title = item['title']
        content = item['content']

        # 构建输入序列：标题 + [SEP] + 内容
        input_text = f"{title}{self.sep_token}{content}"

        # 编码整个序列
        encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,  # 自动添加[CLS]和结尾[SEP]
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 构建标签：标题部分（含[CLS]和标题后的[SEP]）标记为0
        title_tokens = self.tokenizer(
            title + self.sep_token,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )['input_ids']
        title_len = len(title_tokens)

        labels = input_ids.clone()
        labels[:title_len] = 0  # 标题部分不参与损失计算
        labels[input_ids == self.pad_token_id] = 0  # padding部分不参与损失计算

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'title': title,
            'content': content
        }


# 数据加载函数
def load_data(file_path):
    """加载新闻标题-内容数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在：{file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    # 数据清洗
    cleaned_data = []
    for item in data:
        if 'title' in item and 'content' in item and item['title'] and item['content']:
            cleaned_data.append({
                'title': item['title'].strip(),
                'content': item['content'].strip()
            })

    print(f"加载并清洗数据完成，有效样本数：{len(cleaned_data)}")
    return cleaned_data


# 数据分割函数
def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """将数据分割为训练集、验证集和测试集"""
    random.shuffle(data)
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    print(f"数据分割完成 - 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")
    return train_data, val_data, test_data


# 内容生成函数（支持beam search和随机采样）
def generate_content(
        title,
        model,
        tokenizer,
        max_length=256,
        max_gen_len=150,
        beam_size=3,
        temperature=1.0,
        top_k=50
):
    """
    基于标题生成新闻内容
    Args:
        title: 新闻标题
        model: 训练好的模型
        tokenizer: BERT分词器
        max_length: 最大序列长度
        max_gen_len: 最大生成内容长度
        beam_size: beam search的束宽
        temperature: 控制生成多样性（>1增加多样性，<1降低多样性）
        top_k: 仅从概率最高的k个token中采样
    """
    model.eval()
    device = next(model.parameters()).device  # 获取模型所在设备

    # 初始输入：[CLS]标题[SEP]
    input_text = f"{title}{tokenizer.sep_token}"
    input_ids = tokenizer.encode(
        input_text,
        add_special_tokens=True,
        return_tensors='pt'
    ).to(device)

    # 记录初始长度（用于后续截断标题）
    initial_len = input_ids.size(1)

    # 初始化beam：(序列, 累计概率)
    beams = [(input_ids, 0.0)]
    end_token = tokenizer.sep_token_id  # 终止符

    for _ in range(max_gen_len):
        new_beams = []
        for seq, score in beams:
            # 若已达最大长度或生成终止符，直接保留
            if seq.size(1) >= max_length or seq[0, -1].item() == end_token:
                new_beams.append((seq, score))
                continue

            # 模型预测
            with torch.no_grad():
                logits = model(seq)  # (1, seq_len, vocab_size)
                next_logits = logits[0, -1, :]  # 最后一个token的logits

            # 应用temperature调整概率分布
            next_logits = next_logits / temperature

            # 应用top_k过滤
            if top_k > 0:
                top_values, top_indices = torch.topk(next_logits, top_k)
                mask = torch.ones_like(next_logits) * -float('inf')
                mask[top_indices] = 1
                next_logits = next_logits * mask

            # 转换为概率
            next_probs = torch.softmax(next_logits, dim=-1)

            # 取概率最高的beam_size个token
            top_probs, top_ids = torch.topk(next_probs, beam_size)

            # 扩展beam
            for prob, token_id in zip(top_probs, top_ids):
                token_id = token_id.item()
                # 计算累计概率（取对数避免下溢）
                new_score = score + torch.log(prob).item()
                # 拼接新token
                new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                new_beams.append((new_seq, new_score))

        # 保留最优的beam_size个序列
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

        # 如果所有beam都已生成终止符，提前结束
        if all(seq[0, -1].item() == end_token for seq, _ in beams):
            break

    # 选择最优序列（分数最高）
    best_seq = max(beams, key=lambda x: x[1])[0]
    # 解码为文本（跳过特殊符号）
    generated = tokenizer.decode(best_seq[0], skip_special_tokens=True)
    # 截取标题之后的内容
    title_text = tokenizer.decode(tokenizer.encode(title, add_special_tokens=False))
    return generated[len(title_text):].strip()


# 计算BLEU分数（简单实现）
def calculate_bleu(reference, candidate, n_gram=4):
    """计算BLEU分数评估生成内容质量"""
    import re
    # 简单分词
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    ref_tokens = tokenize(reference)
    cand_tokens = tokenize(candidate)

    if len(cand_tokens) == 0:
        return 0.0

    # 计算n-gram准确率
    precisions = []
    for n in range(1, n_gram + 1):
        ref_ngrams = set()
        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams.add(' '.join(ref_tokens[i:i + n]))

        cand_ngrams = []
        for i in range(len(cand_tokens) - n + 1):
            cand_ngrams.append(' '.join(cand_tokens[i:i + n]))

        if len(cand_ngrams) == 0:
            precision = 0.0
        else:
            matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
            precision = matches / len(cand_ngrams)

        precisions.append(precision)

    # 简单平均（实际BLEU使用几何平均和 brevity penalty）
    return sum(precisions) / len(precisions) if precisions else 0.0


# 训练函数
def train(
        data_path,
        pretrain_model_path,
        save_dir='title2content_model',
        epochs=10,
        batch_size=16,
        max_length=256,
        learning_rate=2e-5
):
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # 加载并分割数据
    all_data = load_data(data_path)
    train_data, val_data, test_data = split_data(all_data)

    # 创建数据集和数据加载器
    train_dataset = NewsDataset(train_data, tokenizer, max_length)
    val_dataset = NewsDataset(val_data, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 初始化模型
    model = Title2ContentModel(pretrain_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"使用设备：{device}")

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )

    # 学习率调度器（带预热）
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # 记录最佳验证损失
    best_val_loss = float('inf')

    # 开始训练
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training")

        for batch in progress_bar:
            # 数据移至设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 计算损失
            loss = model(input_ids, attention_mask, labels)
            train_loss += loss.item()

            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪防止爆炸
            optimizer.step()
            scheduler.step()

            # 更新进度条
            progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_bleu = 0.0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # 计算验证损失
                loss = model(input_ids, attention_mask, labels)
                val_loss += loss.item()

                # 随机选择一个样本生成内容并计算BLEU
                if random.random() < 0.1:  # 10%的概率抽样评估
                    idx = random.randint(0, len(batch['title']) - 1)
                    title = batch['title'][idx]
                    true_content = batch['content'][idx]

                    generated_content = generate_content(
                        title, model, tokenizer, max_length
                    )
                    bleu = calculate_bleu(true_content, generated_content)
                    val_bleu += bleu

        avg_val_loss = val_loss / len(val_loader)
        avg_val_bleu = val_bleu / len(val_loader) if len(val_loader) > 0 else 0

        # 打印 epoch 信息
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"验证BLEU分数: {avg_val_bleu:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"最佳模型已保存至：{model_path}")

        # 每轮训练后随机选择测试样本展示生成效果
        sample_idx = random.randint(0, len(val_data) - 1)
        sample_title = val_data[sample_idx]['title']
        sample_true_content = val_data[sample_idx]['content']
        sample_generated = generate_content(sample_title, model, tokenizer, max_length)

        print("\n生成示例:")
        print(f"标题: {sample_title}")
        print(f"真实内容: {sample_true_content[:100]}...")  # 截断显示
        print(f"生成内容: {sample_generated[:100]}...\n")

    # 保存最终模型和分词器
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    tokenizer.save_pretrained(save_dir)
    print(f"最终模型已保存至：{final_model_path}")

    # 测试集评估
    print("\n开始测试集评估...")
    test_bleu_scores = []
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    model.eval()

    for item in tqdm(test_data[:50]):  # 抽样50个测试样本
        generated = generate_content(item['title'], model, tokenizer, max_length)
        bleu = calculate_bleu(item['content'], generated)
        test_bleu_scores.append(bleu)

    print(f"测试集平均BLEU分数: {np.mean(test_bleu_scores):.4f}")

    return model, tokenizer, test_data


# 推理函数（加载已训练模型）
def infer(title, model_dir, pretrain_model_path):
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    # 加载模型
    model = Title2ContentModel(pretrain_model_path)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'best_model.pth'),
        map_location=torch.device('cpu')
    ))
    model.eval()
    # 生成内容
    content = generate_content(title, model, tokenizer)
    return content


if __name__ == "__main__":
    # 配置路径
    DATA_PATH = "sample_data.json"  # 新闻数据文件路径
    PRETRAIN_MODEL_PATH = r"D:\BaiduNetdiskDownload\组件\ppt\AI\数据处理与统计分析\bert-base-chinese"  # 预训练模型
    SAVE_DIR = "./title2content_model"  # 模型保存目录

    # 训练模型
    model, tokenizer, test_data = train(
        data_path=DATA_PATH,
        pretrain_model_path=PRETRAIN_MODEL_PATH,
        save_dir=SAVE_DIR,
        epochs=10,
        batch_size=16,
        max_length=256,
        learning_rate=2e-5
    )

    # 随机选择测试样本进行生成演示
    print("\n===== 测试集生成演示 =====")
    for item in random.sample(test_data, 3):
        title = item['title']
        true_content = item['content']
        generated_content = generate_content(title, model, tokenizer)

        print(f"标题: {title}")
        print(f"真实内容: {true_content}")
        print(f"生成内容: {generated_content}")
        print(f"BLEU分数: {calculate_bleu(true_content, generated_content):.4f}")
        print("-------------------------\n")
