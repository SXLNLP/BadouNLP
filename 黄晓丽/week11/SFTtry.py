import torch
import torch.nn as nn
import torch.utils.data
import json
import random
from transformers import BertTokenizer, BertModel

"""
使用新闻数据尝试实现sft训练
"""

class NewsContentGenerator(nn.Module):
    """新闻内容生成模型，基于BERT架构"""

    def __init__(self, pretrained_model_path):
        """
        初始化模型
        :param pretrained_model_path: 预训练BERT模型路径
        """
        super(NewsContentGenerator, self).__init__()
        # 加载预训练BERT模型作为基础编码器
        self.bert_encoder = BertModel.from_pretrained(pretrained_model_path, return_dict=False)
        # 输出层，预测下一个token的概率分布
        self.output_layer = nn.Linear(768, 21128)  # 21128是BERT中文词表大小
        # 损失函数，忽略填充部分的计算
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, attention_mask=None, target_labels=None):
        """
        模型前向传播
        :param input_ids: 输入token IDs
        :param attention_mask: 注意力掩码（可选）
        :param target_labels: 目标标签（训练时提供）
        :return: 损失值（训练时）或预测概率（生成时）
        """
        # 通过BERT编码器获取上下文表示
        contextual_embeddings, _ = self.bert_encoder(input_ids, attention_mask=attention_mask)
        # 预测每个位置的下一个token
        token_logits = self.output_layer(contextual_embeddings)

        if target_labels is not None:
            # 训练模式：计算预测结果与目标标签之间的损失
            loss = self.loss_fn(token_logits.view(-1, token_logits.shape[-1]),
                                target_labels.view(-1))
            return loss
        else:
            # 生成模式：返回softmax后的概率分布
            return torch.softmax(token_logits, dim=-1)


def load_news_dataset(data_file, max_items=200):
    """
    加载新闻数据集
    :param data_file: 数据文件路径
    :param max_items: 最大加载数量
    :return: 新闻数据列表，每个元素是{'title':标题, 'content':内容}
    """
    news_items = []
    with open(data_file, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            if idx >= max_items:
                break
            if line.strip():
                news_item = json.loads(line.strip())
                news_items.append({
                    'title': news_item['title'],  # 新闻标题
                    'content': news_item['content']  # 新闻内容
                })
    return news_items


def prepare_training_examples(tokenizer, news_items, max_sequence_len=64):
    """
    准备训练样本
    :param tokenizer: 分词器实例
    :param news_items: 新闻数据列表
    :param max_sequence_len: 最大序列长度
    :return: (输入IDs, 标签IDs, 注意力掩码)的元组
    """
    input_ids_list = []
    label_ids_list = []
    attention_masks_list = []

    for item in news_items:
        title = item['title']
        content = item['content']

        # 组合标题和内容，用[SEP]分隔
        combined_text = f"{title}[SEP]{content}"

        # 对文本进行编码
        encoded = tokenizer.encode_plus(
            combined_text,
            max_length=max_sequence_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # 创建标签 - 标题部分设为0（不计算损失），内容部分保留真实ID
        # 找到第一个[SEP]标记的位置
        sep_positions = (input_ids == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            first_sep_idx = sep_positions[0]
            # 标题和第一个SEP标记设为0，内容部分保留
            labels = input_ids.clone()
            labels[:first_sep_idx + 1] = 0
        else:
            # 如果没有找到SEP标记，整个序列设为0
            labels = torch.zeros_like(input_ids)

        input_ids_list.append(input_ids)
        label_ids_list.append(labels)
        attention_masks_list.append(attention_mask)

    return (torch.stack(input_ids_list),
            torch.stack(label_ids_list),
            torch.stack(attention_masks_list))


def initialize_content_generator(pretrained_model_path):
    """
    初始化内容生成器模型
    :param pretrained_model_path: 预训练模型路径
    :return: 模型实例
    """
    model = NewsContentGenerator(pretrained_model_path)
    return model


def generate_content_from_title(title, generator, tokenizer, max_sequence_len=64, max_content_len=80):
    """
    根据标题生成新闻内容
    :param title: 新闻标题
    :param generator: 内容生成模型
    :param tokenizer: 分词器实例
    :param max_sequence_len: 最大序列长度
    :param max_content_len: 生成内容的最大长度
    :return: 生成的新闻内容
    """
    generator.eval()  # 设置为评估模式
    device = next(generator.parameters()).device  # 获取模型所在设备

    # 编码标题并添加[SEP]标记
    input_sequence = tokenizer.encode(
        title + "[SEP]",
        max_length=max_sequence_len,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    # 初始化注意力掩码
    attention_mask = torch.ones_like(input_sequence).to(device)

    # 存储生成的内容tokens
    generated_tokens = []

    # 自回归生成内容
    for _ in range(max_content_len):
        with torch.no_grad():
            # 获取模型预测
            predictions = generator(input_sequence, attention_mask=attention_mask)
            # 获取最后一个位置的token预测
            last_token_probs = predictions[0, -1, :]
            # 选择概率最高的token
            next_token_id = torch.argmax(last_token_probs).item()

            # 遇到[SEP]标记停止生成
            if next_token_id == tokenizer.sep_token_id:
                break

            generated_tokens.append(next_token_id)

            # 更新输入序列
            new_token_tensor = torch.tensor([[next_token_id]]).to(device)
            input_sequence = torch.cat([input_sequence, new_token_tensor], dim=1)
            attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(device)], dim=1)

            # 如果超过最大长度，移除序列开头的token
            if input_sequence.size(1) >= max_sequence_len:
                input_sequence = input_sequence[:, 1:]
                attention_mask = attention_mask[:, 1:]

    # 将生成的token IDs解码为文本
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def train_content_generator(data_file, save_model=True):
    """
    训练新闻内容生成器
    :param data_file: 训练数据文件路径
    :param save_model: 是否保存训练后的模型
    """
    # 训练参数配置
    NUM_EPOCHS = 10  # 训练轮数
    BATCH_SIZE = 8  # 批处理大小
    MAX_SEQ_LENGTH = 64  # 序列最大长度
    LEARNING_RATE = 1e-5  # 学习率
    MAX_SAMPLES = 200  # 最大训练样本数

    # 预训练模型名称
    PRETRAINED_MODEL = r'D:\learn\python39\pythonProject250704\models\bert-base-chinese'

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

    # 加载新闻数据
    news_data = load_news_dataset(data_file, MAX_SAMPLES)
    print(f"已加载 {len(news_data)} 条新闻数据")

    # 准备训练数据
    input_ids, labels, attention_masks = prepare_training_examples(tokenizer, news_data, MAX_SEQ_LENGTH)
    print(f"训练数据准备完成，输入形状: {input_ids.shape}, 标签形状: {labels.shape}")

    # 初始化模型
    content_generator = initialize_content_generator(PRETRAINED_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_generator.to(device)
    print(f"模型已加载到设备: {device}")

    # 设置优化器
    optimizer = torch.optim.AdamW(content_generator.parameters(), lr=LEARNING_RATE)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(input_ids, labels, attention_masks)
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("开始模型训练...")
    for epoch in range(NUM_EPOCHS):
        content_generator.train()  # 设置训练模式
        epoch_total_loss = 0.0

        for batch in data_loader:
            batch_inputs, batch_labels, batch_masks = [tensor.to(device) for tensor in batch]

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播计算损失
            loss = content_generator(batch_inputs,
                                     attention_mask=batch_masks,
                                     target_labels=batch_labels)

            # 反向传播与参数更新
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()

        # 计算并打印本轮平均损失
        avg_loss = epoch_total_loss / len(data_loader)
        print(f"训练轮次 [{epoch + 1}/{NUM_EPOCHS}] - 平均损失: {avg_loss:.4f}")

        # 每轮结束后测试生成效果
        content_generator.eval()
        test_sample = random.choice(news_data)
        title = test_sample['title']
        true_content = test_sample['content']
        generated_content = generate_content_from_title(title, content_generator, tokenizer, MAX_SEQ_LENGTH)

        print("\n当前模型生成效果测试:")
        print(f"标题: {title}")
        print(f"真实内容: {true_content[:60]}..." if len(true_content) > 60 else true_content)
        print(f"生成内容: {generated_content[:60]}..." if len(generated_content) > 60 else generated_content)
        print()

    # 保存训练好的模型
    if save_model:
        model_save_path = "news_content_generator.pth"
        torch.save(content_generator.state_dict(), model_save_path)
        print(f"模型保存位置: {model_save_path}")


if __name__ == "__main__":
    # 开始训练
    train_content_generator("sample_data.json", save_model=True)
