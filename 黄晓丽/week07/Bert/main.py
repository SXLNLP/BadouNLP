# main.py
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from config import global_config as config
from loader import load_and_process_data, analyze_data
from model import BertClassifier
from evaluate import evaluate_model, plot_training_history, predict_sentiment, test_prediction_speed


def main():
    start_time = time.time()
    print("=" * 60)
    print("BERT 情感分类训练")
    print("=" * 60)
    print(f"使用设备: {config.device}")
    print(config)

    # 加载数据
    print("加载数据...")
    train_loader, val_loader = load_and_process_data()

    # 数据分析
    data_stats = analyze_data()
    print("\n数据分析结果:")
    print(f"  好评数量: {data_stats['positive_count']}")
    print(f"  差评数量: {data_stats['negative_count']}")
    print(f"  平均文本长度: {data_stats['avg_length']:.1f} 字符")
    print(f"  最大文本长度: {data_stats['max_length']} 字符")
    print(f"  最小文本长度: {data_stats['min_length']} 字符")

    # 创建模型
    print("\n初始化BERT模型...")
    model = BertClassifier().to(config.device)
    print(model)

    # 优化器 - 使用AdamW（BERT推荐）
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # 损失函数 - 交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print("\n开始训练...")
    train_loss_history = []
    val_acc_history = []
    best_val_acc = 0.0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            token_type_ids = batch['token_type_ids'].to(config.device)
            labels = batch['label'].to(config.device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            optimizer.step()

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # 验证模型
        val_acc, _, _, _ = evaluate_model(model, val_loader, config.device)
        val_acc_history.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.model_save_path)
            print(f"最佳模型已保存至 {config.model_save_path}")

        print(f"Epoch {epoch + 1}/{config.num_epochs} - "
              f"训练损失: {avg_train_loss:.4f} - "
              f"验证准确率: {val_acc:.4f}")

    # 绘制训练曲线
    plot_training_history(train_loss_history, val_acc_history)

    # 加载最佳模型
    model.load_state_dict(torch.load(config.model_save_path))
    model.to(config.device)

    # 最终评估
    final_val_acc, _, _, _ = evaluate_model(model, val_loader, config.device)
    print(f"\n最终验证准确率: {final_val_acc:.4f}")

    # 测试预测性能
    test_samples = [
                       "商品质量非常好，物流也很快，下次还会购买！",
                       "质量太差了，用了一次就坏了，客服也不理人",
                       "价格有点贵，但质量确实不错",
                       "包装简陋，送人拿不出手",
                       "效果一般，没有宣传的那么好"
                   ] * 200  # 重复200次，共1000个样本

    total_time, speed_per_sample = test_prediction_speed(
        model, config.tokenizer, test_samples, config.device, config.max_sequence_length
    )

    print("\n预测性能测试结果:")
    print(f"  总样本数: {len(test_samples)}")
    print(f"  总耗时: {total_time:.4f} 秒")
    print(f"  平均每样本耗时: {speed_per_sample:.4f} 毫秒")
    print(f"  处理速度: {len(test_samples) / total_time:.2f} 样本/秒")

    # 示例预测
    print("\n测试样本预测结果:")
    for sample in test_samples[:5]:  # 只显示前5个
        sentiment, confidence, prob = predict_sentiment(
            sample, model, config.tokenizer, config.device, config.max_sequence_length)
        print(f"文本: '{sample}'")
        print(f"预测: {sentiment} (置信度: {confidence:.2f}, 原始概率: {prob:.4f})")
        print("-" * 60)

    # 生成最终报告
    total_train_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("训练总结报告")
    print("=" * 60)
    print(config)
    print(f"训练总耗时: {total_train_time:.2f} 秒")
    print(f"最终验证准确率: {final_val_acc:.4f}")
    print(f"平均预测延迟: {speed_per_sample:.4f} 毫秒/样本")
    print(f"处理速度: {len(test_samples) / total_time:.2f} 样本/秒")

    # 保存报告
    with open('training_summary.txt', 'w', encoding='utf-8') as f:
        f.write("BERT 训练总结报告\n")
        f.write("=" * 40 + "\n")
        f.write(str(config) + "\n")
        f.write(f"训练总耗时: {total_train_time:.2f} 秒\n")
        f.write(f"最终验证准确率: {final_val_acc:.4f}\n")
        f.write(f"平均预测延迟: {speed_per_sample:.4f} 毫秒/样本\n")
        f.write(f"处理速度: {len(test_samples) / total_time:.2f} 样本/秒\n")


if __name__ == '__main__':
    main()