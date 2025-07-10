# evaluate.py
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # 使用非交互式后端


def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    val_preds = []
    val_true = []
    val_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            # 获取模型输出
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # 获取预测结果
            _, preds = torch.max(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())

    # 计算评估指标
    val_acc = accuracy_score(val_true, val_preds)
    print("\n分类报告:")
    print(classification_report(val_true, val_preds, target_names=['差评', '好评']))

    return val_acc, val_preds, val_true, val_probs


def plot_training_history(train_loss_history, val_acc_history):
    """绘制训练过程曲线"""
    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, 'b-o', label='训练损失')
        plt.title('训练损失变化')
        plt.xlabel('训练轮次')
        plt.ylabel('损失值')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(val_acc_history, 'r-o', label='验证准确率')
        plt.title('验证准确率变化')
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("训练曲线已保存至 training_history.png")

    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")


def predict_sentiment(text, model, tokenizer, device, max_len=128):
    """预测文本情感"""
    try:
        # 使用tokenizer编码文本
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        # 移动到设备
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        # 预测
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(probs, dim=1)
            prob = probs[0][pred].item()
            sentiment = "好评" if pred.item() == 1 else "差评"
            confidence = prob

        return sentiment, confidence, prob

    except Exception as e:
        print(f"预测失败: {e}")
        return "未知", 0.5, 0.5


def test_prediction_speed(model, tokenizer, test_samples, device, max_len=128):
    """测试模型预测速度"""
    start_time = time.perf_counter()

    for sample in test_samples:
        # 编码文本
        encoding = tokenizer.encode_plus(
            sample,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        # 移动到设备
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        # 预测
        model.eval()
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

    total_time = time.perf_counter() - start_time
    num_samples = len(test_samples)
    speed_per_sample = (total_time / num_samples) * 1000  # 毫秒/样本

    return total_time, speed_per_sample