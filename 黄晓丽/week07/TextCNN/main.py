# main.py - 主训练流程

import time
import csv
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from config import global_config as config
from loader import TextDataset
from model import TextCNN
from evaluate import evaluate_model, plot_training_history, predict_sentiment, test_prediction_speed
import matplotlib.pyplot as plt


def main():
    start_time = time.time()
    print("=" * 60)
    print("TextCNN 情感分类训练")
    print("=" * 60)
    print(config)  # 打印配置信息

    # 创建日志文件
    with open(config.log_file, 'w', newline='', encoding='utf-8') as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([
            'epoch', 'train_loss', 'val_acc',
            'epoch_time', 'learning_rate'
        ])

    print("加载数据...")
    data_loader = TextDataset()
    train_loader, val_loader, vocab = data_loader.load_and_process_data()

    # 打印词汇表大小
    print(f"词汇表大小: {config.vocab_size}")

    # 数据分析
    data_stats = data_loader.analyze_data()
    print("\n数据分析结果:")
    print(f"  好评数量: {data_stats['positive_count']}")
    print(f"  差评数量: {data_stats['negative_count']}")
    print(f"  平均文本长度: {data_stats['avg_length']:.1f} 词")
    print(f"  最大文本长度: {data_stats['max_length']} 词")
    print(f"  最小文本长度: {data_stats['min_length']} 词")

    # 创建模型
    print("\n初始化模型...")
    model = TextCNN().to(config.device)
    print(model)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练循环
    print("\n开始训练...")
    train_loss_history = []
    val_acc_history = []
    best_val_acc = 0.0
    epoch_times = []

    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(config.device), labels.to(config.device)

            # 验证输入索引范围
            if torch.max(inputs) >= config.vocab_size:
                print(f"警告: 输入包含无效索引 {torch.max(inputs)} (词汇表大小 {config.vocab_size})")
                # 修正索引
                inputs = torch.clamp(inputs, 0, config.vocab_size - 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })

        # 计算平均训练损失
        train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(train_loss)

        # 验证阶段
        val_acc, _, _, _ = evaluate_model(model, val_loader, config.device)
        val_acc_history.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.model_save_path)

        # 计算本epoch耗时
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # 记录日志
        with open(config.log_file, 'a', newline='', encoding='utf-8') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                epoch + 1, train_loss, val_acc,
                epoch_time, optimizer.param_groups[0]['lr']
            ])

        print(f"Epoch {epoch + 1}/{config.num_epochs} - "
              f"Train Loss: {train_loss:.4f} - "
              f"Val Acc: {val_acc:.4f} - "
              f"Time: {epoch_time:.2f}s")

    # 计算训练总耗时
    total_train_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    # 绘制训练曲线
    plot_training_history(train_loss_history, val_acc_history)

    # 加载最佳模型并测试
    model.load_state_dict(torch.load(config.model_save_path))
    final_val_acc, _, _, _ = evaluate_model(model, val_loader, config.device)
    print(f"\n最终验证集准确率: {final_val_acc:.4f}")

    # 测试预测性能
    _, _, _, samples_per_sec, speed_report = test_prediction_speed(
        model, val_loader, config.predict_test_size, config.device
    )
    print(speed_report)

    # 示例预测
    test_samples = [
        "商品质量非常好，物流也很快，下次还会购买！",
        "质量太差了，用了一次就坏了，客服也不理人",
        "价格有点贵，但质量确实不错",
        "包装简陋，送人拿不出手",
        "效果一般，没有宣传的那么好"
    ]

    # 测试单条预测速度
    single_pred_times = []
    for sample in test_samples:
        start_time = time.perf_counter()
        sentiment, confidence, prob = predict_sentiment(
            sample, model, vocab, config.max_sequence_length, config.device)
        end_time = time.perf_counter()
        pred_time = (end_time - start_time) * 1000  # 毫秒

        single_pred_times.append(pred_time)

        print(f"文本: '{sample}'")
        print(f"预测: {sentiment} (置信度: {confidence:.2f}, 原始概率: {prob:.4f})")
        print(f"预测耗时: {pred_time:.4f} 毫秒")
        print("-" * 60)

    avg_single_time = sum(single_pred_times) / len(single_pred_times)
    print(f"\n平均单条预测耗时: {avg_single_time:.4f} 毫秒")

    # 生成最终报告
    print("\n" + "=" * 60)
    print("训练总结报告")
    print("=" * 60)
    print(config)
    print(f"训练总耗时: {total_train_time:.2f} 秒")
    print(f"平均每轮耗时: {avg_epoch_time:.2f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最终验证准确率: {final_val_acc:.4f}")
    print(f"批量预测速度: {samples_per_sec:.2f} 样本/秒")
    print(f"平均单条预测耗时: {avg_single_time:.4f} 毫秒")

    # 保存报告
    with open('training_summary.txt', 'w', encoding='utf-8') as f:
        f.write("TextCNN 训练总结报告\n")
        f.write("=" * 40 + "\n")
        f.write(str(config) + "\n")
        f.write(f"训练总耗时: {total_train_time:.2f} 秒\n")
        f.write(f"平均每轮耗时: {avg_epoch_time:.2f} 秒\n")
        f.write(f"词汇表大小: {config.vocab_size}\n")
        f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
        f.write(f"最终验证准确率: {final_val_acc:.4f}\n")
        f.write(f"批量预测速度: {samples_per_sec:.2f} 样本/秒\n")
        f.write(f"平均单条预测耗时: {avg_single_time:.4f} 毫秒\n")
        f.write("\n训练历史:\n")
        f.write("Epoch, Train Loss, Val Acc, Time(s), Learning Rate\n")
        with open(config.log_file, 'r', encoding='utf-8') as log:
            f.write(log.read())


if __name__ == '__main__':
    main()