一、基础架构类型对比
模型	          架构类型	   层数	参数量   激活函数
Llama 	      Decoder-Only	80	70B		SwiGLU
Qwen3	      Decoder-Only	96	480B 	GeGLU
DeepSeek-R1	  Decoder-Only	64	671B 	MLA + GeGLU
Mixtral	      MoE-Decoder	32	47B		SwiGLU
DBRX	      MoE-Decoder	64	132B	GeLU
Claude 	      Decoder-Only	-	2T		GeLU
Gemma	      Decoder-Only	28	7B		ReLU

二、注意力机制关键差异
1. 标准优化方案（Llama, Qwen）
# 分组查询注意力 (GQA) - 减少KV缓存
class GroupedQueryAttention(nn.Module):
    def __init__(self, num_heads, num_kv_heads):
        self.q_proj = nn.Linear(d_model, num_heads * d_k)
        self.k_proj = nn.Linear(d_model, num_kv_heads * d_k)  # 共享KV头
        self.v_proj = nn.Linear(d_model, num_kv_heads * d_v)
2. 创新压缩方案（DeepSeek）
# 多头潜在注意力 (MLA) - 低秩KV压缩:cite[10]
class MultiHeadLatentAttention(nn.Module):
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_compressor(self.k_proj(x))  # 压缩至低维
        v = self.v_compressor(self.v_proj(x))
        # 推理时解压缩回原始维度
        k = self.k_decompressor(k) if not training else k
        v = self.v_decompressor(v) if not training else v
        return scaled_dot_product(q, k, v)

3. 混合注意力（GPT-OSS）
# 滑动窗口+全局注意力交替
class HybridAttention(nn.Module):
    def __init__(self):
        self.layers = [
            SlidingWindowAttention(window=1024),  # 局部注意力
            FullAttention()                      # 全局注意力
        ]
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)  # 交替执行
三、归一化与细节优化
1. 归一化层
# 主流方案对比
nn.LayerNorm(d_model)          # GPT-3/4, Claude（标准层归一化）

class RMSNorm(nn.Module):      # Llama, DeepSeek（无中心化）
    def forward(self, x):
        return x * self.weight / torch.sqrt(x.pow(2).mean(-1, keepdim=True))

class DeepNorm(nn.Module):     # ChatGLM（梯度稳定）
    def forward(self, x):
        return x * self.alpha + self.sub_layer(self.norm(x))

2. 前馈网络 (FFN)
# SwiGLU（Llama, Mixtral）
class SwiGLUFFN(nn.Module):
    def forward(self, x):
        x_gate, x_up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(x_gate) * x_up)  # Swish激活
# GeGLU（Qwen3, DBRX）
class GeGLUFFN(nn.Module):
    def forward(self, x):
        return self.down_proj(F.gelu(x_gate) * x_up)  # Gelu激活
