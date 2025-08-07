#week12作业
#看主流模型结构代码，对比结构差异。

#模型	            最独特的设计	                    与其他模型的关键区别
#Baichuan1-13B	    使用Alibi位置编码	                唯一用Alibi的模型，放弃RoPE
#Moss	            平行Transformer结构+GELU_new	    唯一用平行结构、传统FFN、GELU_new的模型
#ChatGLM2	        QKV带bias+Multi-Query	            唯一在QKV用bias的模型，且用Multi-Query
#Llama2	            Multi-Query+全无bias	            最“轻量化”的设计（Multi-Query+无bias）
#Baichuan2-7B	    全主流设计（RoPE+串行+Gated）	    无明显独特性，属于“保守优化”模型
