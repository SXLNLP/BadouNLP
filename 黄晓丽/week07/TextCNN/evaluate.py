# evaluate.py - 模型评估

from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import global_config as config  # 使用全局配置
import jieba
import re
import time


def evaluate_model(model, val_loader, device):
    """评估模型性能"""
    model.eval()
    val_preds = []
    val_true = []
    val_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float().cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())
            val_probs.extend(outputs.cpu().numpy())

    # 计算评估指标
    val_acc = accuracy_score(val_true, val_preds)
    print("\n分类报告:")
    print(classification_report(val_true, val_preds, target_names=['差评', '好评']))

    return val_acc, val_preds, val_true, val_probs


def plot_training_history(train_loss_history, val_acc_history):
    """绘制训练过程曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, 'b-o', label='训练损失')
    plt.title('训练损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_acc_history, 'r-o', label='验证准确率')
    plt.title('验证准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()


def predict_sentiment(text, model, vocab, max_len, device):
    try:
        """预测单个文本情感"""
        if not isinstance(text, str):
            return ""
            # 移除所有非中文字符、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5\w\s，。！？；："\'、]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = list(jieba.cut(text))

        # 转换为索引序列（确保索引在词汇表范围内）
        sequence = []
        for token in tokens:
            # 获取索引，确保在有效范围内
            idx = vocab.get(token, vocab.get('<UNK>', 1))  # 默认为未知词索引

            # 确保索引不超过词汇表大小
            if idx < 0 or idx >= len(vocab):
                idx = vocab.get('<UNK>', 1)
            sequence.append(idx)

        # 截断或填充序列
        if len(sequence) < max_len:
            sequence = sequence + [vocab.get('<PAD>', 0)] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]

        # 转换为张量
        sequence_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)

        print(f"文本: {text}")
        print(f"分词结果: {tokens}")
        print(f"序列长度: {len(sequence)}")
        print(f"最大索引: {max(sequence) if sequence else 0}, 词汇表大小: {len(vocab)}")

        # 预测
        model.eval()
        with torch.no_grad():
            output = model(sequence_tensor)
            prob = output.item()
            sentiment = "好评" if prob > 0.5 else "差评"
            confidence = prob if sentiment == "好评" else 1 - prob

        return sentiment, confidence, prob

    except Exception as e:
        print(f"预测失败: {e}")
        return "未知", 0.5, 0.5

def test_prediction_speed(model, data_loader, num_samples, device):
    """
    测试模型预测速度
    返回:
        total_time: 总预测时间(秒)
        speed_per_sample: 每样本预测时间(毫秒)
        speed_per_batch: 每批次预测时间(毫秒)
        samples_per_sec: 每秒处理的样本数
    """
    model.eval()
    total_time = 0.0
    processed_samples = 0

    # 预热GPU
    if 'cuda' in device:
        warmup_data = next(iter(data_loader))[0][:2].to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(warmup_data)

    # 测试预测速度
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            # 记录开始时间
            start_time = time.perf_counter()

            # 预测
            outputs = model(inputs)

            # 同步GPU操作
            if 'cuda' in device:
                torch.cuda.synchronize()

            # 计算耗时
            batch_time = time.perf_counter() - start_time
            total_time += batch_time
            processed_samples += inputs.size(0)

            # 达到测试样本量后停止
            if processed_samples >= num_samples:
                break

    # 计算性能指标
    speed_per_sample = (total_time / processed_samples) * 1000  # 毫秒/样本
    speed_per_batch = (total_time / (processed_samples / config.batch_size)) * 1000  # 毫秒/批次
    samples_per_sec = processed_samples / total_time

    # 创建性能报告
    speed_report = (
        f"\n预测性能测试 (样本数: {processed_samples}):\n"
        f"  总耗时: {total_time:.4f} 秒\n"
        f"  平均每样本耗时: {speed_per_sample:.4f} 毫秒\n"
        f"  平均每批次耗时: {speed_per_batch:.4f} 毫秒 (批次大小: {config.batch_size})\n"
        f"  处理速度: {samples_per_sec:.2f} 样本/秒\n"
    )

    return total_time, speed_per_sample, speed_per_batch, samples_per_sec, speed_report