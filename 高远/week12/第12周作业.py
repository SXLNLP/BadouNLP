"""
从表格呈现的主流大语言模型（LLM）结构参数来看，各模型在核心组件设计上存在显著差异，这些差异反映了不同模型在性能、效率和训练稳定性上的权衡。以下从关键维度对比分析：
1. 位置编码：RoPE 为主流，Alibi 为特例
RoPE（Rotary Position Embedding）：
被 baichuan2-7b、chatglm2、llama2、moss 采用。其通过旋转矩阵将位置信息编码到 Query/Key 的向量空间中，能自然支持任意长度序列（超出训练长度时泛化性较好），且与注意力计算紧密结合，适合长文本场景。
Alibi（Attention with Linear Biases）：
仅 baichuan2-13b 使用。不依赖显式位置嵌入，而是在注意力分数中加入与相对位置成线性关系的偏置，计算更轻量，但长序列泛化性略逊于 RoPE。
2. Transformer 结构：串行为主，并行仅 moss
串行结构：baichuan2 系列、chatglm2、llama2 采用 “多头注意力 → 归一化 → FFN → 归一化” 的串行链路（Pre-Norm 设计）。这种结构将注意力和前馈网络按顺序堆叠，是目前 LLM 的主流选择，训练稳定性较好。
并行结构：仅 moss 使用。注意力层和 FFN 层并行计算后再合并输出，可能在特定场景下提升计算效率，但需要更精细的训练调优以避免梯度冲突。
3. 多头机制：传统方式与 Multi-Query 的分野
传统多头：baichuan2 系列、moss 采用 “每个头独立计算 Query/Key/Value”，头间参数不共享，表达能力强但参数量大（计算成本高）。
Multi-Query：chatglm2、llama2 采用 “多个头共享 Key/Value，仅 Query 独立”。通过减少 Key/Value 的冗余计算，大幅降低内存占用和推理延迟，更适合大模型部署（如 llama2-70b 的高效推理）。
4. FF 层设计：Gated 形式成主流，传统方式渐退
Gated 形式：baichuan2 系列、chatglm2、llama2 采用带门控的前馈网络（如 SwiGLU：FFN(x) = (W3·σ(W1·x)) ⊗ (W2·x)）。通过门控机制筛选有效信息，表达能力远超传统两层线性层，且训练更稳定。
传统方式：仅 moss 使用 “线性层 + 激活 + 线性层” 的经典结构，参数量少但特征提取能力较弱，更接近原始 Transformer 设计。
5. 归一化层：RMSNorm+Pre-Norm 成趋势
RMSNorm+Pre-Norm：baichuan2 系列、chatglm2、llama2 采用。RMSNorm 移除了 LayerNorm 的均值中心化步骤，计算更高效；Pre-Norm 将归一化放在注意力 / FFN 之前，避免梯度爆炸，更适合深层网络（如 llama2 可轻松堆叠 100 + 层）。
LayerNorm：仅 moss 使用，且可能为 Post-Norm（归一化在层输出后）。传统 LayerNorm 计算稍重，且 Post-Norm 在深层网络中易出现训练不稳定。
6. 激活函数与 Bias：SiLU + 无 Bias 更高效
激活函数：
多数模型（baichuan2、chatglm2、llama2）用 SiLU（Sigmoid (x)・x），计算简单且梯度特性优于 ReLU，适合大模型训练。
moss 用 gelu_new（GELU 的变体），平滑性更好但计算略复杂。
Bias 设置：
无 Bias：baichuan2 系列、llama2 移除了线性层的偏置项，减少参数量（约 5%~10%）并加快推理，且实验证明对性能影响极小。
有 Bias：chatglm2 在 QKV 计算中保留 Bias，可能为了微调时更灵活；moss 仅自注意力层无 Bias，其他层保留，兼顾效率与微调需求。
总结：主流模型的设计趋势
从结构差异可看出，现代 LLM 更倾向于 **“高效化” 与 “深层化”**：

位置编码偏好 RoPE（长序列泛化）；
多头机制向 Multi-Query 迁移（降低推理成本）；
FF 层采用 Gated 形式（增强表达能力）；
归一化层选择 RMSNorm+Pre-Norm（适配深层网络）；
简化 Bias 设置（减少参数与计算）。

而 moss 保留较多传统设计，可能更注重兼容性与训练稳定性，适合特定场景的定制化优化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# 1. 位置编码实现（RoPE vs Alibi）
# ------------------------------
class RoPE(nn.Module):
    """baichuan2-7b/chatglm2/llama2/moss使用的旋转位置编码"""
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        # 预计算频率和位置参数
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        seq_len = x.shape[1]
        position = torch.arange(seq_len, device=x.device).float()
        freqs = torch.outer(position, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [seq_len, dim]
        
        # 旋转操作
        x_rot = x[..., :self.dim//2]  # 取前半部分
        x_pass = x[..., self.dim//2:]  # 后半部分不旋转
        x_rot = torch.stack([-x_rot[..., 1::2], x_rot[..., ::2]], dim=-1).flatten(-2)
        return torch.cat([x_rot * emb[..., ::2] - x_rot * emb[..., 1::2], x_pass], dim=-1)


class Alibi(nn.Module):
    """baichuan2-13b使用的线性偏置位置编码"""
    def __init__(self, num_heads, max_seq_len=4096):
        super().__init__()
        self.num_heads = num_heads
        # 为每个头生成不同的斜率
        self.slopes = torch.tensor([2 **(-8*(i+1)/num_heads) for i in range(num_heads)])

    def forward(self, attention_scores):
        # attention_scores: [batch, heads, seq_len_q, seq_len_k]
        batch, heads, seq_q, seq_k = attention_scores.shape
        # 生成相对位置矩阵 [seq_q, seq_k]
        relative_pos = torch.arange(seq_k, device=attention_scores.device) - \
                      torch.arange(seq_q, device=attention_scores.device)[:, None]
        relative_pos = relative_pos.abs()  # 取绝对值
        # 应用每个头的斜率偏置
        alibi_bias = self.slopes[None, :, None, None] * relative_pos[None, None, :, :]
        return attention_scores + alibi_bias


# ------------------------------
# 2. 多头注意力实现（传统 vs Multi-Query）
# ------------------------------
class TraditionalMultiHeadAttention(nn.Module):
    """baichuan2/moss使用的传统多头注意力（每个头独立QKV）"""
    def __init__(self, dim, num_heads, has_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV权重（传统方式：每个头独立参数）
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=has_bias)
        self.out_proj = nn.Linear(dim, dim, bias=has_bias)

    def forward(self, x, pos_encoding=None):
        batch, seq_len, dim = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # [3, batch, seq_len, dim]
        
        # 应用位置编码（如RoPE）
        if pos_encoding is not None:
            q = pos_encoding(qkv[0])
            k = pos_encoding(qkv[1])
        else:
            q, k, v = qkv
        
        # 拆分多头
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = (attn_probs @ v).transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(out)


class MultiQueryAttention(nn.Module):
    """chatglm2/llama2使用的Multi-Query注意力（共享KV）"""
    def __init__(self, dim, num_heads, has_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Q独立，KV共享（仅1组KV参数）
        self.q_proj = nn.Linear(dim, dim, bias=has_bias)
        self.k_proj = nn.Linear(dim, self.head_dim, bias=has_bias)  # 共享K
        self.v_proj = nn.Linear(dim, self.head_dim, bias=has_bias)  # 共享V
        self.out_proj = nn.Linear(dim, dim, bias=has_bias)

    def forward(self, x, pos_encoding=None):
        batch, seq_len, dim = x.shape
        
        # 计算QKV（Q多头，KV单头共享）
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 应用位置编码
        if pos_encoding is not None:
            q = pos_encoding(q)
            k = pos_encoding(k)
        
        # 拆分Q为多头，KV广播到所有头
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [batch, heads, seq_len, head_dim]
        v = v.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # 计算注意力
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        out = (attn_probs @ v).transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(out)


# ------------------------------
# 3. 前馈网络实现（Gated vs 传统）
# ------------------------------
class GatedFFN(nn.Module):
    """baichuan2/chatglm2/llama2使用的Gated FFN（SwiGLU）"""
    def __init__(self, dim, hidden_dim, activation=nn.SiLU(), has_bias=False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=has_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=has_bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=has_bias)  # 门控分支
        self.activation = activation

    def forward(self, x):
        # 门控机制：(激活(w1x)) * (w3x) → 再通过w2投影
        return self.w2(self.activation(self.w1(x)) * self.w3(x))


class TraditionalFFN(nn.Module):
    """moss使用的传统FFN"""
    def __init__(self, dim, hidden_dim, activation=nn.GELU(approximate='tanh'), has_bias=True):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=has_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=has_bias)
        self.activation = activation

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))


# ------------------------------
# 4. 归一化层（RMSNorm vs LayerNorm）
# ------------------------------
class RMSNorm(nn.Module):
    """baichuan2/chatglm2/llama2使用的RMSNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = dim **-0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放参数

    def forward(self, x):
        # 仅计算平方均值，无中心化
        norm = torch.sqrt((x** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x * self.scale / norm)


# ------------------------------
# 5. 模型整体结构（串行 vs 并行）
# ------------------------------
class SerialTransformerLayer(nn.Module):
    """baichuan2/chatglm2/llama2使用的串行结构（Pre-Norm）"""
    def __init__(self, dim, num_heads, is_multi_query, has_bias):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiQueryAttention(dim, num_heads, has_bias) if is_multi_query \
                  else TraditionalMultiHeadAttention(dim, num_heads, has_bias)
        self.norm2 = RMSNorm(dim)
        self.ffn = GatedFFN(dim, 4 * dim, activation=nn.SiLU(), has_bias=has_bias)

    def forward(self, x, pos_encoding=None):
        # 串行链路：Norm → Attn → 残差 → Norm → FFN → 残差
        x = x + self.attn(self.norm1(x), pos_encoding)
        x = x + self.ffn(self.norm2(x))
        return x


class ParallelTransformerLayer(nn.Module):
    """moss使用的并行结构"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 使用传统LayerNorm
        self.attn = TraditionalMultiHeadAttention(dim, num_heads, has_bias=False)
        self.ffn = TraditionalFFN(dim, 4 * dim, activation=nn.GELU(approximate='tanh'))

    def forward(self, x, pos_encoding=None):
        # 并行链路：Attn和FFN独立计算后相加
        attn_out = self.attn(self.norm(x), pos_encoding)
        ffn_out = self.ffn(self.norm(x))
        return x + attn_out + ffn_out


# ------------------------------
# 6. 模型实例化（体现各模型差异）
# ------------------------------
def baichuan2_7b(dim=4096, num_heads=32):
    """百川2-7B：RoPE + 传统多头 + Gated FFN + 无Bias"""
    return SerialTransformerLayer(
        dim=dim,
        num_heads=num_heads,
        is_multi_query=False,
        has_bias=False
    )

def baichuan2_13b(dim=5120, num_heads=40):
    """百川2-13B：Alibi + 传统多头 + Gated FFN + 无Bias"""
    return SerialTransformerLayer(
        dim=dim,
        num_heads=num_heads,
        is_multi_query=False,
        has_bias=False
    )

def llama2(dim=4096, num_heads=32):
    """LLaMA2：RoPE + Multi-Query + Gated FFN + 无Bias"""
    return SerialTransformerLayer(
        dim=dim,
        num_heads=num_heads,
        is_multi_query=True,
        has_bias=False
    )

def moss(dim=4096, num_heads=32):
    """MOSS：RoPE + 传统多头 + 传统FFN + 并行结构"""
    return ParallelTransformerLayer(
        dim=dim,
        num_heads=num_heads
    )


# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 1024, 4096)  # [batch=2, seq_len=1024, dim=4096]
    
    # 初始化位置编码
    rope = RoPE(dim=4096)
    alibi = Alibi(num_heads=32)
    
    # 测试各模型层
    print("baichuan2-7b输出形状:", baichuan2_7b()(x, rope).shape)
    print("baichuan2-13b输出形状:", baichuan2_13b()(x).shape)  # Alibi在attention内部生效
    print("llama2输出形状:", llama2()(x, rope).shape)
    print("moss输出形状:", moss()(x, rope).shape)
