# 主流大语言模型结构对比分析

## 模型架构核心差异对比

| 模型            | 位置编码  | transformer结构 | 多头机制        | ff层设计   | 归一化层选择           | 激活函数      | 是否使用bias            | 参数规模 | 词汇表大小 |
| ------------- | ----- | ------------- | ----------- | ------- | ---------------- | --------- | ------------------- | ------ | ------- |
| baichuan2-7b  | RoPE  | 串行            | 传统方式        | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 7B     | 125,696 |
| baichuan2-13b | Alibi | 串行            | 传统方式        | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 13B    | 125,696 |
| chatglm2      | RoPE  | 串行            | multi query | gated形式 | RMSnorm/pre norm | SiLU      | qkv有bias，其他线性层无bias | 6B     | 65,024  |
| llama2        | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 7B/13B/70B | 32,000  |
| llama3        | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 8B/70B | 128,256 |
| llama4        | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | qkv有bias，其余无bias    | 405B   | 128,256 |
| moss          | RoPE  | 平行            | 传统方式        | 传统方式    | LayerNorm        | gelu\_new | sa无bias，ff有bias     | 16B    | 106,029 |
| deepseek-v3   | RoPE  | 串行            | GQA         | MoE+gated | RMSnorm/pre norm | SiLU      | 无bias              | 671B   | 129,280 |
| glm-4         | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | qkv有bias，其余无bias    | 9B     | 151,329 |
| qwen2.5       | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 0.5B-72B | 151,936 |
| gemma2        | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | GELU      | 无bias              | 2B/9B/27B | 256,000 |
| mistral       | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 7B/8x7B | 32,000  |
| yi            | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 6B/34B | 64,000  |
| internlm2     | RoPE  | 串行            | GQA         | gated形式 | RMSnorm/pre norm | SiLU      | 无bias              | 1.8B-20B | 92,544  |

