#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import torch
from config import Config
from sft_loader import create_sft_config
from sft_train import main_sft
from sft_evaluate import evaluate_sft_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_complete_sft_pipeline():
    """运行完整的SFT训练流程"""
    
    logger.info("=" * 60)
    logger.info("开始SFT (Supervised Fine-tuning) 训练流程")
    logger.info("=" * 60)
    
    # 步骤1: 检查数据
    data_path = Config["train_data_path"]
    if not os.path.exists(data_path):
        logger.error(f"训练数据文件不存在: {data_path}")
        return False
    
    logger.info(f"✓ 训练数据检查通过: {data_path}")
    
    # 步骤2: 创建SFT配置
    sft_config = create_sft_config(Config)
    
    # 自定义SFT训练参数
    sft_config.update({
        "learning_rate": 1e-4,          # 较小的学习率用于微调
        "epoch": 20,                    # 较少的训练轮数
        "batch_size": 16,               # 可以根据显存调整
        "max_grad_norm": 1.0,           # 梯度裁剪
        "warmup_steps": 50,             # 预热步数
        "response_loss_weight": 1.0,    # 回答部分损失权重
    })
    
    # 如果存在预训练模型，使用它作为起点
    pretrained_path = "output/epoch_200.pth"
    if os.path.exists(pretrained_path):
        sft_config["pretrained_model_path"] = pretrained_path
        logger.info(f"✓ 找到预训练模型: {pretrained_path}")
    else:
        logger.info("⚠ 未找到预训练模型，将从头开始训练")
    
    # 步骤3: 开始SFT训练
    try:
        logger.info("开始SFT训练...")
        model = main_sft(sft_config)
        logger.info("✓ SFT训练完成")
    except Exception as e:
        logger.error(f"✗ SFT训练失败: {e}")
        return False
    
    # 步骤4: 评估模型
    best_model_path = os.path.join(sft_config["model_path"], "best_sft_model.pth")
    if os.path.exists(best_model_path):
        try:
            logger.info("开始评估SFT模型...")
            accuracy = evaluate_sft_model(best_model_path, sft_config, logger)
            logger.info(f"✓ SFT模型评估完成，准确率: {accuracy:.2%}")
        except Exception as e:
            logger.error(f"✗ 模型评估失败: {e}")
    
    logger.info("=" * 60)
    logger.info("SFT训练流程完成！")
    logger.info("=" * 60)
    
    # 打印使用说明
    print_usage_instructions()
    
    return True

def print_usage_instructions():
    """打印使用说明"""
    print("\n" + "=" * 60)
    print("SFT训练完成！以下是模型文件说明：")
    print("=" * 60)
    print("output/")
    print("   ├── best_sft_model.pth     # 最佳SFT模型（推荐使用）")
    print("   ├── final_sft_model.pth    # 最终SFT模型")
    print("   ├── sft_epoch_*.pth        # SFT训练检查点")
    print("   └── epoch_200.pth          # 原始预训练模型（如果存在）")
    print()
    print("如何使用训练好的SFT模型：")
    print("1. 评估模型: python sft_evaluate.py")
    print("2. 交互式测试: 运行sft_evaluate.py中的interactive_test方法")
    print("3. 批量推理: 参考sft_evaluate.py中的代码")
    print()
    print("SFT vs 原始模型的区别：")
    print("• SFT模型: 经过指令微调，能更好地理解和遵循指令")
    print("• 原始模型: 标准seq2seq训练，直接content->title映射")
    print("• SFT优势: 更好的指令理解、更稳定的输出格式")
    print("=" * 60)

def create_demo_data():
    """创建演示数据（如果需要的话）"""
    demo_data = [
        {
            "title": "北京今日天气晴朗，最高气温25度",
            "content": "据北京气象台消息，今日北京天气晴朗，万里无云，最高气温将达到25摄氏度，微风习习，是外出游玩的好天气。市民可以适当减少衣物，但早晚温差较大，请注意保暖。"
        },
        {
            "title": "科技公司发布新款智能手机",
            "content": "某知名科技公司今日正式发布了其最新款智能手机，该手机采用了最新的处理器技术，电池续航能力大幅提升，相机性能也有显著改进。预计将于下月正式上市销售。"
        },
        {
            "title": "教育部宣布新的教育改革措施",
            "content": "教育部今日宣布了一系列新的教育改革措施，包括减轻学生课业负担、提高教师待遇、改进教学方法等。这些措施旨在促进教育公平，提高教育质量，培养学生的创新能力。"
        }
    ]
    
    import json
    with open("demo_data.json", "w", encoding="utf8") as f:
        for item in demo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info("✓ 创建演示数据文件: demo_data.json")

def check_environment():
    """检查运行环境"""
    logger.info("检查运行环境...")
    
    # 检查Python版本
    import sys
    if sys.version_info < (3, 6):
        logger.error("Python版本需要3.6以上")
        return False
    
    # 检查PyTorch
    try:
        import torch
        logger.info(f"✓ PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA可用，设备数量: {torch.cuda.device_count()}")
        else:
            logger.info("⚠ CUDA不可用，将使用CPU训练")
    except ImportError:
        logger.error("PyTorch未安装")
        return False
    
    # 检查必要文件
    required_files = ["vocab.txt", "sample_data.json"]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"必需文件不存在: {file_path}")
            return False
        logger.info(f"✓ 文件检查通过: {file_path}")
    
    return True

if __name__ == "__main__":
    # 检查环境
    if not check_environment():
        logger.error("环境检查失败，请检查依赖和文件")
        sys.exit(1)
    
    # 如果没有演示数据，创建一些
    if not os.path.exists("sample_data.json") or os.path.getsize("sample_data.json") < 100:
        logger.info("创建演示数据...")
        create_demo_data()
    
    # 运行完整的SFT流程
    success = run_complete_sft_pipeline()
    
    if success:
        logger.info("SFT训练流程成功完成！")
        sys.exit(0)
    else:
        logger.error("SFT训练流程失败")
        sys.exit(1)