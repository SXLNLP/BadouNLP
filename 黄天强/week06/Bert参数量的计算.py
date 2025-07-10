import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertModel


class DiyBert(nn.Module):
    def __init__(self, state_dict, num_layers=12):
        super().__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = num_layers
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # Embeddings
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"]
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"]
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"]
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"]
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"]

        # Transformer layers
        self.transformer_weights = []
        for i in range(self.num_layers):
            weights = [
                state_dict[f"encoder.layer.{i}.attention.self.query.weight"],
                state_dict[f"encoder.layer.{i}.attention.self.query.bias"],
                state_dict[f"encoder.layer.{i}.attention.self.key.weight"],
                state_dict[f"encoder.layer.{i}.attention.self.key.bias"],
                state_dict[f"encoder.layer.{i}.attention.self.value.weight"],
                state_dict[f"encoder.layer.{i}.attention.self.value.bias"],
                state_dict[f"encoder.layer.{i}.attention.output.dense.weight"],
                state_dict[f"encoder.layer.{i}.attention.output.dense.bias"],
                state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"],
                state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"],
                state_dict[f"encoder.layer.{i}.intermediate.dense.weight"],
                state_dict[f"encoder.layer.{i}.intermediate.dense.bias"],
                state_dict[f"encoder.layer.{i}.output.dense.weight"],
                state_dict[f"encoder.layer.{i}.output.dense.bias"],
                state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"],
                state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"]
            ]
            self.transformer_weights.append(weights)

        # Pooler
        self.pooler_dense_weight = state_dict["pooler.dense.weight"]
        self.pooler_dense_bias = state_dict["pooler.dense.bias"]

    def embedding_forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        token_type_ids = torch.zeros_like(input_ids)

        word_emb = self.word_embeddings[input_ids]
        pos_emb = self.position_embeddings[pos_ids]
        token_type_emb = self.token_type_embeddings[token_type_ids]

        embedding = word_emb + pos_emb + token_type_emb
        return self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)

    def single_transformer_layer_forward(self, x, weights):
        q_w, q_b, k_w, k_b, v_w, v_b, \
            attention_out_w, attention_out_b, \
            attn_ln_w, attn_ln_b, \
            inter_w, inter_b, \
            out_w, out_b, \
            ff_ln_w, ff_ln_b = weights

        attention_out = self.self_attention(
            x, q_w, q_b, k_w, k_b, v_w, v_b,
            attention_out_w, attention_out_b,
            self.num_attention_heads, self.hidden_size
        )
        x = self.layer_norm(x + attention_out, attn_ln_w, attn_ln_b)

        ff_out = self.feed_forward(x, inter_w, inter_b, out_w, out_b)
        x = self.layer_norm(x + ff_out, ff_ln_w, ff_ln_b)
        return x

    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b,
                       out_w, out_b, num_heads, hidden_size):
        # x: [batch_size, seq_len, hidden_size]
        q = torch.matmul(x, q_w.T) + q_b
        k = torch.matmul(x, k_w.T) + k_b
        v = torch.matmul(x, v_w.T) + v_b

        batch_size, seq_len, _ = q.size()
        head_dim = hidden_size // num_heads
        q = self.transpose_for_scores(q, num_heads, head_dim)  # [batch, heads, seq_len, head_dim]
        k = self.transpose_for_scores(k, num_heads, head_dim)
        v = self.transpose_for_scores(v, num_heads, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)
        # context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)

        return torch.matmul(context, out_w.T) + out_b

    def feed_forward(self, x, inter_w, inter_b, out_w, out_b):
        x = F.gelu(torch.matmul(x, inter_w.T) + inter_b)
        return torch.matmul(x, out_w.T) + out_b

    def transpose_for_scores(self, x, num_heads, head_dim):
        batch_size, seq_len, hidden_size = x.size()
        x = x.view(batch_size, seq_len, num_heads, head_dim)
        return x.transpose(1, 2)

    def layer_norm(self, x, w, b, eps=1e-12):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, unbiased=False, keepdim=True)
        return (x - mean) / (std + eps) * w + b

    def pooler_output_layer(self, x_cls):
        x = torch.matmul(x_cls, self.pooler_dense_weight.T) + self.pooler_dense_bias
        return torch.tanh(x)

    def forward(self, input_ids):
        x = self.embedding_forward(input_ids)
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, self.transformer_weights[i])
        pooled = self.pooler_output_layer(x[:, 0])
        return x, pooled


class TransformerLayer(nn.Module):
    def __init__(self, state_dict, i, hidden_size=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        prefix = f"encoder.layer.{i}"

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)
        self.attn_norm = nn.LayerNorm(hidden_size)

        self.inter = nn.Linear(hidden_size, hidden_size * 4)
        self.out = nn.Linear(hidden_size * 4, hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size)

        # Load weights
        self.q.weight.data.copy_(state_dict[f"{prefix}.attention.self.query.weight"])
        self.q.bias.data.copy_(state_dict[f"{prefix}.attention.self.query.bias"])
        self.k.weight.data.copy_(state_dict[f"{prefix}.attention.self.key.weight"])
        self.k.bias.data.copy_(state_dict[f"{prefix}.attention.self.key.bias"])
        self.v.weight.data.copy_(state_dict[f"{prefix}.attention.self.value.weight"])
        self.v.bias.data.copy_(state_dict[f"{prefix}.attention.self.value.bias"])
        self.attn_out.weight.data.copy_(state_dict[f"{prefix}.attention.output.dense.weight"])
        self.attn_out.bias.data.copy_(state_dict[f"{prefix}.attention.output.dense.bias"])
        self.attn_norm.weight.data.copy_(state_dict[f"{prefix}.attention.output.LayerNorm.weight"])
        self.attn_norm.bias.data.copy_(state_dict[f"{prefix}.attention.output.LayerNorm.bias"])
        self.inter.weight.data.copy_(state_dict[f"{prefix}.intermediate.dense.weight"])
        self.inter.bias.data.copy_(state_dict[f"{prefix}.intermediate.dense.bias"])
        self.out.weight.data.copy_(state_dict[f"{prefix}.output.dense.weight"])
        self.out.bias.data.copy_(state_dict[f"{prefix}.output.dense.bias"])
        self.ffn_norm.weight.data.copy_(state_dict[f"{prefix}.output.LayerNorm.weight"])
        self.ffn_norm.bias.data.copy_(state_dict[f"{prefix}.output.LayerNorm.bias"])

    def transpose_for_scores(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).reshape(x.size(0), x.size(1), self.hidden_size)
        x = self.attn_norm(x + self.attn_out(context))

        ff = self.out(F.gelu(self.inter(x)))
        x = self.ffn_norm(x + ff)
        return x


class DiyBert2(nn.Module):
    def __init__(self, state_dict, num_layers=12, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding layers
        vocab_size = state_dict["embeddings.word_embeddings.weight"].size(0)
        max_position_embeddings = state_dict["embeddings.position_embeddings.weight"].size(0)
        type_vocab_size = state_dict["embeddings.token_type_embeddings.weight"].size(0)

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.embeddings_layer_norm = nn.LayerNorm(hidden_size)

        self.word_embeddings.weight.data.copy_(state_dict["embeddings.word_embeddings.weight"])
        self.position_embeddings.weight.data.copy_(state_dict["embeddings.position_embeddings.weight"])
        self.token_type_embeddings.weight.data.copy_(state_dict["embeddings.token_type_embeddings.weight"])
        self.embeddings_layer_norm.weight.data.copy_(state_dict["embeddings.LayerNorm.weight"])
        self.embeddings_layer_norm.bias.data.copy_(state_dict["embeddings.LayerNorm.bias"])

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerLayer(state_dict, i, hidden_size, num_heads)
            for i in range(num_layers)
        ])

        # Pooler
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.pooler.weight.data.copy_(state_dict["pooler.dense.weight"])
        self.pooler.bias.data.copy_(state_dict["pooler.dense.bias"])

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_type_ids = torch.zeros_like(input_ids)

        x = self.word_embeddings(input_ids) + \
            self.position_embeddings(pos_ids) + \
            self.token_type_embeddings(token_type_ids)

        x = self.embeddings_layer_norm(x)

        for layer in self.layers:
            x = layer(x)

        pooled = torch.tanh(self.pooler(x[:, 0]))  # [CLS] token
        return x, pooled


def count_bert_parameters():
    # BERT-base 配置参数
    vocab_size = 21128              # 词汇表大小（例如中文 BERT 使用 21128 个词）
    hidden_size = 768               # 每个 token 的隐藏层维度（BERT-base 使用 768）
    intermediate_size = 3072        # 前馈网络的中间层维度（通常是 hidden_size 的 4 倍）
    max_position_embeddings = 512   # 支持的最大序列长度
    type_vocab_size = 2             # 类型（segment）词表大小，支持句子 A/B
    num_hidden_layers = 12          # Transformer 层数（BERT-base 是 12 层）

    # 1. Embedding 层参数数量
    embeddings_total = (
            vocab_size * hidden_size +                  # token embeddings：词汇表大小 × hidden_size
            max_position_embeddings * hidden_size +     # position embeddings：位置数 × hidden_size
            type_vocab_size * hidden_size +             # segment embeddings：类型数量 × hidden_size
            2 * hidden_size                             # LayerNorm 的 gamma 和 beta，各是 hidden_size
    )

    # 2. Transformer 每层的参数（以下为单层参数）
    self_attention_qkv = 3 * (hidden_size * hidden_size + hidden_size)          # 自注意力中的 Q、K、V
    attention_output = hidden_size * hidden_size + hidden_size                  # 自注意力输出层
    attention_layernorm = 2 * hidden_size                                       # 注意力后的 LayerNorm

    intermediate_dense = hidden_size * intermediate_size + intermediate_size    # 前馈网络的第一层
    output_dense = intermediate_size * hidden_size + hidden_size                # 前馈网络的第二层
    ffn_layernorm = 2 * hidden_size                                             # 前馈网络之后的 LayerNorm

    transformer_layer_params = (                                                # 计算单层 Transformer 层的总参数
            self_attention_qkv +
            attention_output +
            attention_layernorm +
            intermediate_dense +
            output_dense +
            ffn_layernorm
    )

    # 所有 Transformer 层的参数（共 12 层）
    transformer_total = num_hidden_layers * transformer_layer_params

    # 3. Pooler 层参数（用于 [CLS] 向量的变换）
    pooler_total = hidden_size * hidden_size + hidden_size                      # 全连接层

    # 总参数量 = Embedding 层 + 所有 Transformer 层 + Pooler 层
    total_params = embeddings_total + transformer_total + pooler_total

    return total_params


if __name__ == "__main__":
    x = torch.tensor([2450, 15486, 102, 2110]).unsqueeze(0)  # Shape: [1, seq_len]

    # Load pre-trained model
    bert = BertModel.from_pretrained("../bert-base-chinese", return_dict=False)
    bert.eval()

    # Run torch BERT
    seq_output, pooled_output = bert(x)
    print("测试官方模型输出：")
    print(seq_output.shape, ',', pooled_output.shape)
    print("sequence output:", seq_output[0, 0, :5])
    print("pooled output:", pooled_output[0, :5])
    print()

    # Run custom BERT
    diy_model = DiyBert(bert.state_dict())  # 没有 grad_fn，这些只是普通的参数，不属于模型的可训练的参数
    diy_model.eval()
    seq_output, pooled_output = diy_model(x)
    print("测试diy模型1输出：")
    print(seq_output.shape, ',', pooled_output.shape)
    print("sequence output:", seq_output[0, 0, :5])
    print("pooled output:", pooled_output[0, :5])
    print()

    # Run custom BERT 2
    diy_model2 = DiyBert2(bert.state_dict())  # 可训练
    diy_model2.eval()
    seq_output, pooled_output = diy_model2(x)
    print("测试diy模型2输出：")
    print(seq_output.shape, ',', pooled_output.shape)
    print("sequence output:", seq_output[0, 0, :5])
    print("pooled output:", pooled_output[0, :5])
    print()

    actual_param_count = sum(p.numel() for p in bert.state_dict().values())
    estimated_param_count = count_bert_parameters()
    print(f"实际模型参数个数: {actual_param_count:,}")
    print(f"手动估算参数个数: {estimated_param_count:,}")
    print(f"diy模型1参数个数: {sum(p.numel() for p in diy_model.parameters()):,}")
    print(f"diy模型2参数个数: {sum(p.numel() for p in diy_model2.parameters()):,}")
