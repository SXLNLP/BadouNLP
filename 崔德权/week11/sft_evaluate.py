# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import jieba

from config import Config
from sft_model import NewsTitleGenerator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTEvaluator:
    """SFT模型评估器"""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.config = config
        self.generator = NewsTitleGenerator(model_path, config)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1
    
    def load_test_data(self, data_path: str) -> List[Dict[str, str]]:
        """加载测试数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'title' in item and 'content' in item:
                        data.append({
                            'title': item['title'].strip(),
                            'content': item['content'].strip()
                        })
                except json.JSONDecodeError:
                    continue
        return data
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """计算BLEU分数"""
        # 使用jieba分词
        ref_tokens = list(jieba.cut(reference))
        cand_tokens = list(jieba.cut(candidate))
        
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smooth)
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """计算ROUGE分数"""
        scores = self.scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def evaluate(self, test_data: List[Dict[str, str]], max_samples: int = None) -> Dict[str, Any]:
        """评估模型性能"""
        if max_samples:
            test_data = test_data[:max_samples]
        
        logger.info(f"Evaluating on {len(test_data)} samples...")
        
        bleu_scores = []
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        generated_titles = []
        reference_titles = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            content = item['content']
            reference_title = item['title']
            
            # 生成标题
            generated_title = self.generator.generate_title(content)
            
            # 计算BLEU分数
            bleu_score = self.calculate_bleu(reference_title, generated_title)
            bleu_scores.append(bleu_score)
            
            # 计算ROUGE分数
            rouge_score = self.calculate_rouge(reference_title, generated_title)
            for key in rouge_scores:
                rouge_scores[key].append(rouge_score[key])
            
            generated_titles.append(generated_title)
            reference_titles.append(reference_title)
        
        # 计算平均分数
        avg_bleu = np.mean(bleu_scores)
        avg_rouge = {key: np.mean(scores) for key, scores in rouge_scores.items()}
        
        results = {
            'bleu': avg_bleu,
            'rouge': avg_rouge,
            'generated_titles': generated_titles,
            'reference_titles': reference_titles,
            'individual_scores': {
                'bleu': bleu_scores,
                'rouge': rouge_scores
            }
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """打印评估结果"""
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        logger.info(f"BLEU Score: {results['bleu']:.4f}")
        logger.info(f"ROUGE-1 Score: {results['rouge']['rouge1']:.4f}")
        logger.info(f"ROUGE-2 Score: {results['rouge']['rouge2']:.4f}")
        logger.info(f"ROUGE-L Score: {results['rouge']['rougeL']:.4f}")
        logger.info("=" * 50)
        
        # 打印一些示例
        logger.info("\nSAMPLE PREDICTIONS:")
        logger.info("-" * 50)
        for i in range(min(5, len(results['generated_titles'])):
            logger.info(f"Content: {results['reference_titles'][i][:50]}...")
            logger.info(f"Reference: {results['reference_titles'][i]}")
            logger.info(f"Generated: {results['generated_titles'][i]}")
            logger.info("-" * 30)
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """保存评估结果"""
        # 移除生成和参考标题以减小文件大小
        save_results = {
            'bleu': results['bleu'],
            'rouge': results['rouge'],
            'individual_scores': results['individual_scores']
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {save_path}")

def main():
    """主函数"""
    # 模型路径
    model_path = "./checkpoints/best_model"  # 根据实际路径调整
    
    # 创建评估器
    evaluator = SFTEvaluator(model_path, Config)
    
    # 加载测试数据
    test_data = evaluator.load_test_data(Config["train_data_path"])
    
    # 评估模型
    results = evaluator.evaluate(test_data, max_samples=100)  # 限制样本数量
    
    # 打印结果
    evaluator.print_results(results)
    
    # 保存结果
    evaluator.save_results(results, "evaluation_results.json")

if __name__ == "__main__":
    main() 