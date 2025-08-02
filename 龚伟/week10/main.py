import os
import torch
import numpy as np
import logging
from config import Config
from evaluate import Evaluator
from loader import load_data
from transformers import BertModel  # 引入Bert模型
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义Bert+生成头模型
class BertTitleGenerator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.hidden_size = self.bert.config.hidden_size
        # 关键修改：直接使用Bert模型的词汇表大小，而非config
        self.generator = torch.nn.Linear(self.hidden_size, self.bert.config.vocab_size)
        # 自回归mask矩阵（下三角，确保仅用历史token）
        self.register_buffer("ar_mask", self._create_ar_mask(config["output_max_length"]))

    def _create_ar_mask(self, max_len):
        """生成自回归mask：下三角为1（可见），上三角为0（不可见）"""
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)  # (1,1,max_len,max_len)
        return mask  # 用于目标序列的自注意力

    # 在BertTitleGenerator类的forward方法中，修改Bert编码部分
    def forward(self, input_ids, attention_mask, target_ids):
        # 1. 用Bert编码新闻内容（输入）
        # 旧版本transformers中，BertModel返回元组：(last_hidden_state, pooler_output, ...)
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 从元组中提取最后一层隐藏状态（第一个元素）
        content_emb = bert_output[0]  # 替换原代码的 bert_output.last_hidden_state

        # 2. 编码目标序列（标题前缀，用于自回归）
        target_emb = self.bert.embeddings(target_ids)  # 仅用Bert的词嵌入层

        # 3. 目标序列自注意力（仅关注历史token）
        target_attn_mask = self.ar_mask[:, :, :target_ids.shape[1], :target_ids.shape[1]]
        # 复用Bert的第一个编码器层（注意：旧版本可能返回元组，需提取第一个元素）
        target_self_attn = self.bert.encoder.layer[0](
            hidden_states=target_emb,
            attention_mask=target_attn_mask
        )[0]  # 从元组中提取隐藏状态

        # 4. 融合内容编码和目标编码（简化为仅用目标序列）
        fused_emb = target_self_attn

        # 5. 预测下一个token
        logits = self.generator(fused_emb)  # (batch, target_len, vocab_size)
        return logits

# 选择优化器（Bert推荐用AdamW）
def choose_optimizer(config, model):
    # 固定Bert参数，仅微调生成头（可选，视数据量而定）
    for param in model.bert.parameters():
        param.requires_grad = False  # 小数据集时固定Bert，仅训练生成头
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01
    )
    return optimizer

def main(config):
    os.makedirs(config["model_path"], exist_ok=True)
    # 初始化模型
    model = BertTitleGenerator(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
        logger.info("使用GPU训练")
    # 数据和优化器
    train_data = load_data(config["train_data_path"], config, logger)
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    # 损失函数（忽略PAD的loss）
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=config["pad_idx"])

    # 训练循环
    best_rouge = 0  # 早停用
    for epoch in range(config["epoch"]):
        model.train()
        train_loss = []
        for batch in train_data:
            if cuda_flag:
                batch = {k: v.cuda() for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            target_ids = batch["target_ids"]
            gold_ids = batch["gold_ids"]

            # 前向传播
            logits = model(input_ids, attention_mask, target_ids)
            loss = loss_func(logits.transpose(1, 2), gold_ids)  # (batch, vocab, len) vs (batch, len)

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())

        # 打印训练损失
        avg_loss = np.mean(train_loss)
        logger.info(f"Epoch {epoch+1} | 平均损失: {avg_loss:.4f}")

        # 验证并早停
        current_rouge = evaluator.eval(epoch+1)
        if current_rouge > best_rouge:
            best_rouge = current_rouge
            torch.save(model.state_dict(), os.path.join(config["model_path"], "best_model.pth"))
            logger.info(f"保存最佳模型（ROUGE: {best_rouge:.4f}）")

if __name__ == "__main__":
    main(Config)