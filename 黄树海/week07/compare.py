import os
import torch
import csv
import logging
from config import Config
from evaluate import Evaluator
from model import TorchModel
from loader import load_data
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建保存结果的 CSV 文件
result_file = "model_test_results3.csv"
with open(result_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "model_type", "learning_rate", "hidden_size",
        "batch_size", "pooling_style", "acc", "100_ref_time"
    ])

    # 获取所有 .pth 模型文件
    model_base_path = "output"
    model_files = []
    target_model_num = 12
    for root, dirs, files in os.walk(model_base_path):
        for file in files:
            if file.endswith(str(target_model_num)+".pth"):
                model_files.append(os.path.join(root, file))

    # 加载测试数据（shuffle=False 确保每次取的是相同的数据）
    test_data = load_data(Config["valid_data_path"], Config, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("模型文件有：")
    print(len(model_files))
    # 遍历每个模型并测试
    for model_path in model_files:
        print(f"\nLoading model from {model_path}")

        # 解析模型参数
        path_parts = model_path.split(os.sep)
        try:
            model_type = path_parts[1]
            learning_rate = float(path_parts[2])
            hidden_size = int(path_parts[3])
            batch_size = int(path_parts[4])
            pooling_style = path_parts[5]
        except Exception as e:
            print("路径解析失败，请确认模型路径格式正确:", e)
            continue

        # 构建模型结构
        config = Config.copy()
        config["model_type"] = model_type
        config["learning_rate"] = learning_rate
        config["hidden_size"] = hidden_size
        config["batch_size"] = batch_size
        config["pooling_style"] = pooling_style

        model = TorchModel(config)
        model.to(device)

        # 加载权重
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print("模型加载完毕，开始测试")
        # 创建 evaluator
        evaluator = Evaluator(config, model, logger)  # 如果不需要 logger，可以传 None

        # 调用 eval 方法获取准确率
        accuracy,avg_time = evaluator.eval(epoch=target_model_num)

        total_time = avg_time*100

        # 写入 CSV
        writer.writerow([
            model_type,
            learning_rate,
            hidden_size,
            batch_size,
            pooling_style,
            f"{accuracy:.4f}",
            f"{total_time:.6f}"
        ])

        print(f"模型 {model_type} 测试完成")
        print(f"准确率: {accuracy:.4f}, 100条总耗时: {total_time:.6f} 秒")

print(f"\n测试结果已保存至 {result_file}")