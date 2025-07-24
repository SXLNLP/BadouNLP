# main.py
import time
import pandas as pd
import numpy as np
from config import global_config as config
from loader import TextDataset
from evaluate import evaluate_classifier, plot_evaluation, predict_sentiment, test_prediction_speed
import joblib


def main():
    start_time = time.time()
    print("=" * 60)
    print("FastText (Gensim) 情感分类系统")
    print("=" * 60)
    print(config)

    # 创建数据集处理器
    data_processor = TextDataset()

    # 数据分析
    data_stats = data_processor.analyze_data()
    print("\n数据分析结果:")
    print(f"  好评数量: {data_stats['positive_count']}")
    print(f"  差评数量: {data_stats['negative_count']}")
    print(f"  平均文本长度: {data_stats['avg_length']:.1f} 字符")
    print(f"  最大文本长度: {data_stats['max_length']} 字符")
    print(f"  最小文本长度: {data_stats['min_length']} 字符")

    # 训练词向量和分类器
    train_start = time.time()
    train_df, val_df, X_train, y_train, X_val, y_val = data_processor.load_and_process_data()
    train_time = time.time() - train_start

    # 评估分类器
    accuracy, report = evaluate_classifier(data_processor.classifier, X_val, y_val)

    # 绘制评估结果
    plot_evaluation(accuracy)

    # 测试预测性能
    test_samples = [
                       "商品质量非常好，物流也很快，下次还会购买！",
                       "质量太差了，用了一次就坏了，客服也不理人",
                       "价格有点贵，但质量确实不错",
                       "包装简陋，送人拿不出手",
                       "效果一般，没有宣传的那么好"
                   ] * 200  # 重复200次，共1000个样本

    # 测试预测速度
    pred_start = time.time()
    total_time, speed_per_sample = test_prediction_speed(
        test_samples, data_processor.vector_model, data_processor.classifier)
    pred_time = time.time() - pred_start

    # 输出性能报告
    print("\n预测性能测试结果:")
    print(f"  总样本数: {len(test_samples)}")
    print(f"  总耗时: {total_time:.4f} 秒")
    print(f"  平均每样本耗时: {speed_per_sample:.4f} 毫秒")
    print(f"  处理速度: {len(test_samples) / total_time:.2f} 样本/秒")

    # 示例预测
    print("\n测试样本预测结果:")
    for sample in test_samples[:5]:  # 只显示前5个
        sentiment, confidence, prob = predict_sentiment(
            sample, data_processor.vector_model, data_processor.classifier)
        print(f"文本: '{sample}'")
        print(f"预测: {sentiment} (置信度: {confidence:.2f}, 原始概率: {prob:.4f})")
        print("-" * 60)

    # 生成最终报告
    print("\n" + "=" * 60)
    print("训练总结报告")
    print("=" * 60)
    print(config)
    print(f"训练总耗时: {train_time:.2f} 秒")
    print(f"验证准确率: {accuracy:.4f}")
    print(f"预测测试耗时: {pred_time:.2f} 秒")
    print(f"平均预测延迟: {speed_per_sample:.4f} 毫秒/样本")
    print(f"处理速度: {len(test_samples) / total_time:.2f} 样本/秒")

    # 保存报告
    with open('training_summary.txt', 'w', encoding='utf-8') as f:
        f.write("FastText (Gensim) 训练总结报告\n")
        f.write("=" * 40 + "\n")
        f.write(str(config) + "\n")
        f.write(f"训练总耗时: {train_time:.2f} 秒\n")
        f.write(f"验证准确率: {accuracy:.4f}\n")
        f.write(f"预测测试耗时: {pred_time:.2f} 秒\n")
        f.write(f"平均预测延迟: {speed_per_sample:.4f} 毫秒/样本\n")
        f.write(f"处理速度: {len(test_samples) / total_time:.2f} 样本/秒\n")
        f.write("\n分类报告:\n")
        f.write(report)


if __name__ == '__main__':
    main()