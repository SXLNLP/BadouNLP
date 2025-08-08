模型	pre/post/sandwich -LN	LN类型	Group Query	Attention位置编码	MOE	共享专家	MLP结构	激活函数	transformer	 attention是否有偏置
grok1(马斯克)	sandwich	RMSNorm	yes	RoPE	yes	no	llama2	gelu	串行	无
deepseekv3	pre	RMSNorm	yes	RoPE	yes	yes	llama2	silu	串行	无
llama4（Meta）	pre	RMSNorm	yes	RoPE	yes	yes	llama2	gelu	串行	无
glm4	sandwich	RMSNorm	yes	RoPE	no	no	llama2	silu	串行	有
gpt_neox	post	LayerNorm	no	RoPE	no	no	传统模式	gelu	串行/并行	有
qwen3	pre	RMSNorm	yes	RoPE	no	no	llama2	silu	串行	无
gemma3	sandwich	RMSNorm	yes	RoPE	no	no	llama2	gelu_pytorch_tanh	串行	无
openai	post	LayerNorm		无	no	no	传统模式	gelu	串行	无
