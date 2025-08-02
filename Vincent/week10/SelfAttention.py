import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.modeling_bert import BertSelfAttention, BertModel

class CausalSelfAttention(BertSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask_to_apply = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            attention_mask_to_apply = attention_mask

        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 因果 mask（只对 self-attention 生效）
        if not is_cross_attention:
            L = attention_scores.size(-1)
            causal_mask = torch.tril(torch.ones((L, L), device=attention_scores.device, dtype=torch.bool))
            attention_scores = attention_scores.masked_fill(~causal_mask, float("-inf"))

        if attention_mask_to_apply is not None:
            attention_scores = attention_scores + attention_mask_to_apply

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)
        return outputs


def make_bert_mask(bert: BertModel) -> BertModel:
    cfg = bert.config  # 用模型全局的 config，而不是 old_sa.config
    for i, layer in enumerate(bert.encoder.layer):
        old_sa = layer.attention.self
        new_sa = CausalSelfAttention(cfg)

        # BertSdpaSelfAttention 的 state_dict 键名和 BertSelfAttention 基本一致，但以防万一用 strict=False
        try:
            new_sa.load_state_dict(old_sa.state_dict(), strict=False)
        except Exception as e:
            print(f"[WARN] layer {i} load_state_dict strict=False fallback, err = {e}")
            sd = new_sa.state_dict()
            old_sd = old_sa.state_dict()
            for k in sd.keys():
                if k in old_sd and sd[k].shape == old_sd[k].shape:
                    sd[k].copy_(old_sd[k])
            new_sa.load_state_dict(sd, strict=False)

        layer.attention.self = new_sa
    return bert
