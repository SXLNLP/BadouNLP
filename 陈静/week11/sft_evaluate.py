# -*- coding: utf-8 -*-
import torch
import collections
import io
import json
import six
import sys
import argparse
from sft_loader import load_sft_data
from collections import defaultdict, OrderedDict

from transformer.Translator import Translator

"""
SFT模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_sft_data(config["valid_data_path"], config, logger, shuffle=False)
        self.reverse_vocab = dict([(y, x) for x, y in self.valid_data.dataset.vocab.items()])
        self.translator = Translator(
            self.model,
            config["beam_size"],
            config["output_max_length"],
            config["pad_idx"],
            config["pad_idx"],
            config["start_idx"],
            config["end_idx"]
        )
        
        # SFT指令模板（与训练时保持一致）
        self.instruction_templates = [
            "请为以下新闻内容生成合适的标题：",
            "根据以下新闻内容，生成一个准确的新闻标题：",
            "请阅读以下新闻并提供简洁的标题：",
            "为这篇新闻写一个恰当的标题：",
            "请总结以下新闻内容并生成标题："
        ]

    def eval(self, epoch):
        """评估SFT模型效果"""
        self.logger.info("开始测试第%d轮SFT模型效果：" % epoch)
        self.model.eval()
        self.model.cpu()
        self.stats_dict = defaultdict(int)
        
        total_samples = 0
        correct_predictions = 0
        
        for index, batch_data in enumerate(self.valid_data):
            input_seqs, target_seqs, gold = batch_data
            
            for i, input_seq in enumerate(input_seqs):
                if total_samples >= 10:  # 只评估前10个样本
                    break
                    
                # 生成标题
                generate = self.translator.translate_sentence(input_seq.unsqueeze(0))
                
                # 解码序列
                input_text = self.decode_seq(input_seq)
                generated_text = self.decode_seq(generate)
                target_text = self.decode_seq(target_seqs[i])
                
                # 打印结果
                print(f"\n=== 样本 {total_samples + 1} ===")
                print("输入：", input_text.replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", ""))
                print("生成：", generated_text.replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", ""))
                print("标准：", target_text.replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", ""))
                
                # 简单的准确性评估
                if self.simple_match(generated_text, target_text):
                    correct_predictions += 1
                
                total_samples += 1
                
            if total_samples >= 10:
                break
        
        # 计算准确率
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        self.logger.info(f"SFT模型评估完成，准确率: {accuracy:.2%} ({correct_predictions}/{total_samples})")
        
        return accuracy

    def decode_seq(self, seq):
        """解码序列为文本"""
        return "".join([self.reverse_vocab[int(idx)] for idx in seq])
    
    def simple_match(self, generated, target):
        """简单的匹配评估"""
        # 去除特殊标记
        gen_clean = generated.replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
        tgt_clean = target.replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
        
        # 计算简单的字符匹配度
        if len(tgt_clean) == 0:
            return False
        
        common_chars = sum(1 for c in gen_clean if c in tgt_clean)
        similarity = common_chars / len(tgt_clean)
        
        return similarity > 0.5  # 50%以上匹配度认为正确
    
    def interactive_test(self):
        """交互式测试SFT模型"""
        self.logger.info("进入SFT模型交互式测试模式")
        self.model.eval()
        self.model.cpu()
        
        print("\n=== SFT模型交互式测试 ===")
        print("输入新闻内容，系统将生成标题（输入'quit'退出）：")
        
        while True:
            try:
                news_content = input("\n请输入新闻内容: ").strip()
                if news_content.lower() == 'quit':
                    break
                
                if not news_content:
                    continue
                
                # 添加指令
                instruction = self.instruction_templates[0]  # 使用第一个模板
                full_input = instruction + news_content
                
                # 编码输入
                input_ids = self.encode_text(full_input)
                input_tensor = torch.LongTensor(input_ids).unsqueeze(0)
                
                # 生成标题
                generate = self.translator.translate_sentence(input_tensor)
                generated_title = self.decode_seq(generate)
                
                # 清理输出
                clean_title = generated_title.replace("[PAD]", "").replace("[CLS]", "").replace("[SEP]", "")
                
                print(f"生成的标题: {clean_title}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"生成出错: {e}")
        
        print("退出交互式测试")
    
    def encode_text(self, text):
        """编码文本为token ids"""
        input_id = []
        for char in text:
            input_id.append(self.valid_data.dataset.vocab.get(char, self.valid_data.dataset.vocab["[UNK]"]))
        
        # 截断或填充
        max_length = self.config["input_max_length"]
        input_id = input_id[:max_length]
        input_id += [self.config["pad_idx"]] * (max_length - len(input_id))
        
        return input_id

def evaluate_sft_model(model_path, config, logger):
    """评估指定的SFT模型"""
    from transformer.Models import Transformer
    
    # 加载模型
    model = Transformer(
        config["vocab_size"], config["vocab_size"], 0, 0,
        d_word_vec=128, d_model=128, d_inner=256,
        n_layers=1, n_head=2, d_k=64, d_v=64,
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # 创建评估器
    evaluator = SFTEvaluator(config, model, logger)
    
    # 运行评估
    accuracy = evaluator.eval(0)
    
    # 交互式测试
    evaluator.interactive_test()
    
    return accuracy

if __name__ == "__main__":
    import logging
    from config import Config
    from sft_loader import create_sft_config
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建SFT配置
    sft_config = create_sft_config(Config)
    
    # 评估最佳SFT模型
    best_model_path = "output/best_sft_model.pth"
    if os.path.exists(best_model_path):
        logger.info(f"评估SFT模型: {best_model_path}")
        accuracy = evaluate_sft_model(best_model_path, sft_config, logger)
        logger.info(f"最终评估准确率: {accuracy:.2%}")
    else:
        logger.error(f"SFT模型文件不存在: {best_model_path}")