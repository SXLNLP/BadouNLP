# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer  # 需安装：pip install rouge-score
from transformers import BertTokenizer
import torch.nn.functional as F  # 添加这行：导入F

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.valid_data = load_data(config["valid_data_path"], config, logger, shuffle=False)
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=False)  # 中文无需stemmer

    def eval(self, epoch):
        self.logger.info(f"开始评估第{epoch}轮模型")
        self.model.eval()
        total_rouge1 = []
        total_rougeL = []
        with torch.no_grad():
            for batch in self.valid_data:
                input_ids = batch["input_ids"]
                gold_ids = batch["gold_ids"]
                # 生成标题
                generated_ids = self.generate_title(input_ids)
                # 解码为文本
                for gen_ids, gold_id in zip(generated_ids, gold_ids):
                    gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    gold_text = self.tokenizer.decode(gold_id, skip_special_tokens=True)
                    # 计算ROUGE
                    scores = self.scorer.score(gold_text, gen_text)
                    total_rouge1.append(scores['rouge1'].fmeasure)
                    total_rougeL.append(scores['rougeL'].fmeasure)
                    # 打印示例（每10个样本打印1个）
                    if len(total_rouge1) % 10 == 0:
                        self.logger.info(f"\n输入内容: {self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[:50]}...")
                        self.logger.info(f"生成标题: {gen_text}")
                        self.logger.info(f"真实标题: {gold_text}")
        # 平均指标
        avg_r1 = np.mean(total_rouge1)
        avg_rL = np.mean(total_rougeL)
        self.logger.info(f"ROUGE-1: {avg_r1:.4f}, ROUGE-L: {avg_rL:.4f}")
        return avg_rL  # 用ROUGE-L作为早停指标

    def generate_title(self, input_ids):
        """自回归生成标题（带N-gram惩罚避免重复）"""
        batch_size = input_ids.shape[0]
        generated = torch.full((batch_size, 1), self.config["start_idx"], dtype=torch.long)  # 初始：[CLS]
        if torch.cuda.is_available():
            generated = generated.cuda()
            input_ids = input_ids.cuda()

        for _ in range(self.config["output_max_length"] - 1):
            # 停止条件：所有样本都已生成[SEP]
            if (generated == self.config["end_idx"]).all(dim=1).all():
                break
            # 前向传播预测下一个token
            logits = self.model(
                input_ids=input_ids,
                attention_mask=(input_ids != self.config["pad_idx"]).float(),
                target_ids=generated
            )  # (batch, seq_len, vocab)
            next_logits = logits[:, -1, :]  # 取最后一个位置的预测

            # N-gram惩罚：抑制重复的2-gram
            next_logits = self.ngram_penalty(generated, next_logits)

            # 采样（或beam search）：这里用top-k采样避免贪心导致的重复
            next_token = self.top_k_sampling(next_logits, k=10)
            # 拼接生成结果
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        return generated

    def ngram_penalty(self, generated, logits):
        """N-gram惩罚：若2-gram已出现，降低对应token概率"""
        batch_size = generated.shape[0]
        for i in range(batch_size):
            # 提取已生成的2-gram
            gen_seq = generated[i].cpu().numpy().tolist()
            ngrams = set()
            for j in range(len(gen_seq)-1):
                ngram = (gen_seq[j], gen_seq[j+1])
                ngrams.add(ngram)
            # 若下一个token与最后一个token组成的2-gram已存在，惩罚
            last_token = gen_seq[-1]
            for token in range(logits.shape[1]):
                if (last_token, token) in ngrams:
                    logits[i, token] /= self.config["repetition_penalty"]  # 降低概率
        return logits

    def top_k_sampling(self, logits, k=10):
        """top-k采样：从概率最高的k个token中随机选择（增加多样性）"""
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        # 重新归一化
        top_k_probs = top_k_probs / torch.sum(top_k_probs, dim=-1, keepdim=True)
        # 随机选择
        selected = torch.multinomial(top_k_probs, 1).squeeze(1)
        return torch.gather(top_k_indices, 1, selected.unsqueeze(1)).squeeze(1)

# 复用loader中的load_data
from loader import load_data