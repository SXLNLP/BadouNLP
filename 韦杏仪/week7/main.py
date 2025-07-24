import torch
import os
import numpy as np
from config import config
from loader import load_data, create_data_loaders
from model import get_model
from evaluate import train_model, evaluate_speed
import pandas as pd
from tabulate import tabulate

# 设置随机种子，确保结果可复现
torch.manual_seed(config.seed)
np.random.seed(config.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
X_train, X_val, y_train, y_val = load_data()

# 创建数据加载器和词表
train_loader, val_loader, vocab = create_data_loaders(X_train, X_val, y_train, y_val)

# 存储模型对比结果
results = []

# 遍历要训练的模型类型
for model_type in config.models:
    print(f"\n{'=' * 50}")
    print(f"开始训练 {model_type} 模型")
    print(f"{'=' * 50}")

    # 获取模型
    model = get_model(model_type, len(vocab))
    print(f"模型结构:\n{model}\n")

    # 训练模型
    model = model.to(device)  # 将模型移到GPU/CPU
    train_history, best_model = train_model(model, train_loader, val_loader, len(vocab), device)

    # 评估模型预测速度
    model.load_state_dict(best_model)  # 加载最佳模型参数
    infer_speed = evaluate_speed(model, val_loader, device)

    # 记录结果
    best_val_acc = max(train_history['val_acc'])
    results.append({
        '模型类型': model_type,
        '最佳验证准确率': f"{best_val_acc * 100:.2f}%",
        '预测速度(样本/秒)': f"{infer_speed:.2f}"
    })

    # 保存模型
    model_path = os.path.join(config.save_dir, f"{model_type}_model.pth")
    torch.save(best_model, model_path)
    print(f"{model_type} 模型已保存到: {model_path}")

# 打印模型对比结果
print("\n\n===== 模型对比结果 =====")
if results:
    # 使用tabulate库创建美观的表格
    print(tabulate(results, headers='keys', tablefmt='grid'))

    # 保存结果到CSV文件
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(config.save_dir, 'model_comparison.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"结果已保存到: {results_csv}")
else:
    print("没有模型训练完成")

print("\n训练全部完成！")
