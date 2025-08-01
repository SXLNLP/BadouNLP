# -*- coding: utf-8 -*-

"""
SFT训练启动脚本
快速开始新闻标题生成的SFT训练
"""

import os
import sys
import argparse
from config import Config

def main():
    parser = argparse.ArgumentParser(description='SFT新闻标题生成器启动脚本')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'inference'], 
                       default='train', help='运行模式')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model',
                       help='模型路径（用于评估和推理）')
    parser.add_argument('--data_path', type=str, default='sample_data.json',
                       help='数据文件路径')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not os.path.exists(args.data_path):
        print(f"错误：数据文件 {args.data_path} 不存在")
        print("请确保数据文件存在且格式正确")
        return
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        print("开始SFT训练...")
        print(f"数据文件: {args.data_path}")
        print(f"配置: {Config}")
        
        # 更新数据路径
        Config["train_data_path"] = args.data_path
        Config["valid_data_path"] = args.data_path
        
        # 导入并运行训练
        from sft_train import main as train_main
        train_main()
        
    elif args.mode == 'evaluate':
        if not os.path.exists(args.model_path):
            print(f"错误：模型路径 {args.model_path} 不存在")
            print("请先训练模型或指定正确的模型路径")
            return
        
        print("开始模型评估...")
        from sft_evaluate import main as eval_main
        eval_main()
        
    elif args.mode == 'inference':
        if not os.path.exists(args.model_path):
            print(f"错误：模型路径 {args.model_path} 不存在")
            print("请先训练模型或指定正确的模型路径")
            return
        
        print("启动推理模式...")
        import subprocess
        subprocess.run([
            sys.executable, 'sft_inference.py',
            '--model_path', args.model_path,
            '--mode', 'interactive'
        ])

if __name__ == "__main__":
    main() 