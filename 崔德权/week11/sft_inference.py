# -*- coding: utf-8 -*-

import torch
import json
from typing import List, Dict, Any
import argparse

from config import Config
from sft_model import NewsTitleGenerator

class SFTInference:
    """SFT模型推理类"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.generator = NewsTitleGenerator(model_path, config)
    
    def generate_title(self, content: str, max_length: int = 128) -> str:
        """生成单个标题"""
        return self.generator.generate_title(content, max_length)
    
    def batch_generate(self, contents: List[str], max_length: int = 128) -> List[str]:
        """批量生成标题"""
        return self.generator.batch_generate(contents, max_length)
    
    def interactive_mode(self):
        """交互模式"""
        print("=" * 50)
        print("新闻标题生成器 - 交互模式")
        print("输入新闻内容，模型将生成标题")
        print("输入 'quit' 退出")
        print("=" * 50)
        
        while True:
            try:
                content = input("\n请输入新闻内容: ").strip()
                
                if content.lower() == 'quit':
                    print("再见！")
                    break
                
                if not content:
                    print("请输入有效的新闻内容")
                    continue
                
                # 生成标题
                title = self.generate_title(content)
                print(f"\n生成的标题: {title}")
                
            except KeyboardInterrupt:
                print("\n\n程序被中断，再见！")
                break
            except Exception as e:
                print(f"生成过程中出现错误: {e}")
    
    def process_file(self, input_file: str, output_file: str):
        """处理文件中的新闻内容"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f]
            
            contents = [item['content'] for item in data]
            titles = self.batch_generate(contents)
            
            # 保存结果
            results = []
            for i, (item, title) in enumerate(zip(data, titles)):
                result = {
                    'id': i,
                    'content': item['content'],
                    'original_title': item.get('title', ''),
                    'generated_title': title
                }
                results.append(result)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            print(f"处理完成，结果已保存到 {output_file}")
            
        except Exception as e:
            print(f"处理文件时出现错误: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SFT新闻标题生成器')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model',
                       help='模型路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'file'], default='interactive',
                       help='运行模式：interactive(交互模式) 或 file(文件处理模式)')
    parser.add_argument('--input_file', type=str, help='输入文件路径')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--content', type=str, help='单条新闻内容')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = SFTInference(args.model_path, Config)
    
    if args.mode == 'interactive':
        inference.interactive_mode()
    elif args.mode == 'file':
        if not args.input_file or not args.output_file:
            print("文件处理模式需要指定 --input_file 和 --output_file")
            return
        inference.process_file(args.input_file, args.output_file)
    elif args.content:
        # 处理单条内容
        title = inference.generate_title(args.content)
        print(f"新闻内容: {args.content}")
        print(f"生成的标题: {title}")

if __name__ == "__main__":
    main() 