# 作业描述：其路径下的‘对比.xls’文件


# 防丢补充其内内容
# 模型名称	位置编码	transformer结构	多头机制	ff层设计	归一化层选择	激活函数	是否使用bias
# baichuan2-7b	RoPE	串行	传统方式	gated形式	RMSnorm/pre norm	SiLU	无bias
# baichuan2-13b	Alibi	串行	传统方式	gated形式	RMSnorm/pre norm	SiLU	无bias
# chatglm2	RoPE	串行	multi query	gated形式	RMSnorm/pre norm	SiLU	qkv有bias，其他线性层无bias
# llama2	RoPE	串行	multi query	gated形式	RMSnorm/pre norm	SiLU	无bias
# moss	RoPE	平行	传统方式	传统方式	LayerNorm	gelu_new	sa无bias, ff有bias
# DBRX	RoPE	串行	GQA（分组查询）	MoE（gated）	RMSnorm/pre norm	SiLU	无bias
# deepseek	RoPE	串行	传统方式	gated形式	RMSnorm/pre norm	SiLU	无bias
# gemma	RoPE	串行	multi query	gated形式	RMSnorm/pre norm	GeLU	无bias
# grok1	RoPE	串行	multi query	MoE（gated）	RMSnorm/pre norm	SiLU	无bias
# Mixtral	RoPE	串行	GQA（分组查询）	MoE（gated）	RMSnorm/pre norm	SiLU	无bias
# Qwen-7B 	RoPE	串行	传统方式	gated形式	RMSnorm/pre norm	SiLU	无bias
