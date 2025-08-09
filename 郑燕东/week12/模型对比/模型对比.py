名称	位置编码	transformer结构	多头机制	ff层设计	归一化层选择	激活函数	是否使用bias
gpt	正弦位置编码	平行	传统方式	传统方式	norm	gelu	有bias
deepseek	RoPE	串行	Grouped query	gated形式	RMSnorm	SiLU	无bias
qwen3	RoPE	串行	传统方式	GLU	RMSnorm	gelu	无bias
lama	RoPE	串行	Grouped query	gated形式	RMSnorm	SiLU	无bias
moss	RoPE	平行	传统方式	传统方式	LayerNorm	gelu_new	sa无bias, ff有bias
gemma	RoPE	串行	Grouped query	GLU	RMSnorm	gelu	无bias
