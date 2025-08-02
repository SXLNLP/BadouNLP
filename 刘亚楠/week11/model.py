from transformers import BertLMHeadModel, BertConfig
from config import config as train_config
import torch
import torch.nn.functional as F


class Title2ContentModel(BertLMHeadModel):
    def __init__(self, config=None):  # 关键修改：添加默认参数
        # 自动提供默认config
        if config is None:
            config = BertConfig.from_pretrained(train_config.pretrained_model)
        super().__init__(config)

    def forward(self, input_ids, attention_mask, loss_mask=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        if loss_mask is not None:
            # 确保获取logits
            logits = outputs.logits if isinstance(outputs, dict) else outputs[0]

            # 移一位以预测下一个token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # 确保loss_mask是连续的并移一位
            loss_mask = loss_mask.contiguous()
            loss_mask = loss_mask[..., 1:].reshape(-1)  # 使用reshape代替view

            # 计算损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )

            # 应用mask并计算平均损失
            masked_loss = loss * loss_mask
            total_loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)  # 避免除零

            outputs.loss = total_loss

        return outputs