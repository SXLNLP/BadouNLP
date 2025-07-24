import torch
import math
import numpy as np
from transformers import BertModel

'''
通过手动矩阵运算实现BERT结构，并添加训练过程参数形状跟踪
模型文件下载 https://huggingface.co/models
'''


# softmax归一化
def softmax(x):
    print(f"Softmax input shape: {x.shape}" if hasattr(x, 'shape') else f"Softmax input shape: {np.shape(x)}")
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # 数值稳定性处理
    result = e_x / np.sum(e_x, axis=-1, keepdims=True)
    print(f"Softmax output shape: {result.shape}" if hasattr(result,
                                                             'shape') else f"Softmax output shape: {np.shape(result)}")
    return result


# gelu激活函数
def gelu(x):
    print(f"GELU input shape: {x.shape}" if hasattr(x, 'shape') else f"GELU input shape: {np.shape(x)}")
    result = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))
    print(
        f"GELU output shape: {result.shape}" if hasattr(result, 'shape') else f"GELU output shape: {np.shape(result)}")
    return result


class DiyBert:
    # 将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1  # 注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights(state_dict)
        self.training = False  # 添加训练模式标志
        self.optimizer = None  # 优化器占位符
        self.loss_fn = None  # 损失函数占位符

    def load_weights(self, state_dict):
        # embedding部分
        print("Loading embedding weights...")
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        print(f"word_embeddings shape: {self.word_embeddings.shape}")

        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        print(f"position_embeddings shape: {self.position_embeddings.shape}")

        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        print(f"token_type_embeddings shape: {self.token_type_embeddings.shape}")

        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        print(f"embeddings_layer_norm_weight shape: {self.embeddings_layer_norm_weight.shape}")

        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        print(f"embeddings_layer_norm_bias shape: {self.embeddings_layer_norm_bias.shape}")

        self.transformer_weights = []
        # transformer部分，有多层
        print("\nLoading transformer weights...")
        for i in range(self.num_layers):
            print(f"\nLoading weights for layer {i}...")
            q_w = state_dict[f"encoder.layer.{i}.attention.self.query.weight"].numpy()
            q_b = state_dict[f"encoder.layer.{i}.attention.self.query.bias"].numpy()
            k_w = state_dict[f"encoder.layer.{i}.attention.self.key.weight"].numpy()
            k_b = state_dict[f"encoder.layer.{i}.attention.self.key.bias"].numpy()
            v_w = state_dict[f"encoder.layer.{i}.attention.self.value.weight"].numpy()
            v_b = state_dict[f"encoder.layer.{i}.attention.self.value.bias"].numpy()
            attention_output_weight = state_dict[f"encoder.layer.{i}.attention.output.dense.weight"].numpy()
            attention_output_bias = state_dict[f"encoder.layer.{i}.attention.output.dense.bias"].numpy()
            attention_layer_norm_w = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy()
            attention_layer_norm_b = state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy()
            intermediate_weight = state_dict[f"encoder.layer.{i}.intermediate.dense.weight"].numpy()
            intermediate_bias = state_dict[f"encoder.layer.{i}.intermediate.dense.bias"].numpy()
            output_weight = state_dict[f"encoder.layer.{i}.output.dense.weight"].numpy()
            output_bias = state_dict[f"encoder.layer.{i}.output.dense.bias"].numpy()
            ff_layer_norm_w = state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy()
            ff_layer_norm_b = state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy()

            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b,
                                             attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b,
                                             intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])

            # 打印各层参数形状
            print(f"Layer {i} parameters shapes:")
            print(f"  q_w: {q_w.shape}, q_b: {q_b.shape}")
            print(f"  k_w: {k_w.shape}, k_b: {k_b.shape}")
            print(f"  v_w: {v_w.shape}, v_b: {v_b.shape}")
            print(
                f"  attention_output_weight: {attention_output_weight.shape}, attention_output_bias: {attention_output_bias.shape}")
            print(
                f"  attention_layer_norm_w: {attention_layer_norm_w.shape}, attention_layer_norm_b: {attention_layer_norm_b.shape}")
            print(f"  intermediate_weight: {intermediate_weight.shape}, intermediate_bias: {intermediate_bias.shape}")
            print(f"  output_weight: {output_weight.shape}, output_bias: {output_bias.shape}")
            print(f"  ff_layer_norm_w: {ff_layer_norm_w.shape}, ff_layer_norm_b: {ff_layer_norm_b.shape}")

        # pooler层
        print("\nLoading pooler weights...")
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        print(f"pooler_dense_weight shape: {self.pooler_dense_weight.shape}")
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()
        print(f"pooler_dense_bias shape: {self.pooler_dense_bias.shape}")

    # bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        print("\nEmbedding forward pass:")
        print(f"Input shape: {x.shape}")
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shape: [max_len, hidden_size]
        print(f"Word embeddings output shape: {we.shape}")

        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings,
                                np.array(list(range(len(x)))))  # shape: [max_len, hidden_size]
        print(f"Position embeddings output shape: {pe.shape}")

        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shape: [max_len, hidden_size]
        print(f"Token type embeddings output shape: {te.shape}")

        embedding = we + pe + te
        print(f"Summed embeddings shape: {embedding.shape}")

        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight,
                                    self.embeddings_layer_norm_bias)  # shape: [max_len, hidden_size]
        print(f"Layer norm output shape: {embedding.shape}")
        return embedding

    # embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        print(f"Getting embeddings for input shape: {x.shape}")
        result = np.array([embedding_matrix[index] for index in x])
        print(f"Embedding result shape: {result.shape}")
        return result

    # 执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        print("\nAll transformer layers forward pass:")
        print(f"Input shape: {x.shape}")
        for i in range(self.num_layers):
            print(f"\nTransformer layer {i} forward pass:")
            x = self.single_transformer_layer_forward(x, i)
        return x

    # 执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        # 取出该层的参数
        q_w, q_b, \
            k_w, k_b, \
            v_w, v_b, \
            attention_output_weight, attention_output_bias, \
            attention_layer_norm_w, attention_layer_norm_b, \
            intermediate_weight, intermediate_bias, \
            output_weight, output_bias, \
            ff_layer_norm_w, ff_layer_norm_b = weights

        # self attention层
        print(f"\nSelf-attention forward pass:")
        attention_output = self.self_attention(x,
                                               q_w, q_b,
                                               k_w, k_b,
                                               v_w, v_b,
                                               attention_output_weight, attention_output_bias,
                                               self.num_attention_heads,
                                               self.hidden_size)
        print(f"Self-attention output shape: {attention_output.shape}")

        # bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        print(f"After attention layer norm and residual shape: {x.shape}")

        # feed forward层
        print(f"\nFeed forward forward pass:")
        feed_forward_x = self.feed_forward(x,
                                           intermediate_weight, intermediate_bias,
                                           output_weight, output_bias)
        print(f"Feed forward output shape: {feed_forward_x.shape}")

        # bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        print(f"After feed forward layer norm and residual shape: {x.shape}")

        return x

    # self attention的计算
    def self_attention(self,
                       x,
                       q_w,
                       q_b,
                       k_w,
                       k_b,
                       v_w,
                       v_b,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        print("\nSelf-attention computation:")
        print(f"Input shape: {x.shape}")
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]
        print(f"Q matrix shape: {q.shape}")

        k = np.dot(x, k_w.T) + k_b  # shape: [max_len, hidden_size]
        print(f"K matrix shape: {k.shape}")

        v = np.dot(x, v_w.T) + v_b  # shape: [max_len, hidden_size]
        print(f"V matrix shape: {v.shape}")

        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        print(f"Reshaped Q shape: {q.shape}")

        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        print(f"Reshaped K shape: {k.shape}")

        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        print(f"Reshaped V shape: {v.shape}")

        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))
        print(f"Attention scores shape: {qk.shape}")

        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        print(f"Softmax attention weights shape: {qk.shape}")

        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        print(f"Context vectors shape: {qkv.shape}")

        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        print(f"Reshaped context vectors shape: {qkv.shape}")

        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        print(f"Final attention output shape: {attention.shape}")
        return attention

    # 多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        print(f"\nTransposing for multi-head attention:")
        print(f"Input shape: {x.shape}")
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        print(f"Output shape: {x.shape}")
        return x

    # 前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        print("\nFeed forward computation:")
        print(f"Input shape: {x.shape}")
        # output shape: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        print(f"Intermediate layer output shape: {x.shape}")

        x = gelu(x)
        print(f"After GELU activation shape: {x.shape}")

        # output shape: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        print(f"Final feed forward output shape: {x.shape}")
        return x

    # 归一化层
    def layer_norm(self, x, w, b):
        print(f"\nLayer normalization:")
        print(f"Input shape: {x.shape}")
        mean = np.mean(x, axis=1, keepdims=True)
        var = np.var(x, axis=1, keepdims=True)
        x = (x - mean) / np.sqrt(var + 1e-12)  # 添加小常数防止除以0
        x = x * w + b
        print(f"Output shape: {x.shape}")
        return x

    # 链接[cls] token的输出层
    def pooler_output_layer(self, x):
        print("\nPooler output layer:")
        print(f"Input shape: {x.shape}")
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        print(f"Dense layer output shape: {x.shape}")
        x = np.tanh(x)
        print(f"Final pooler output shape: {x.shape}")
        return x

    # 最终输出
    def forward(self, x):
        print("\nBERT model forward pass:")
        print(f"Input shape: {x.shape}")
        x = self.embedding_forward(x)
        print(f"After embedding shape: {x.shape}")
        sequence_output = self.all_transformer_layer_forward(x)
        print(f"Final sequence output shape: {sequence_output.shape}")
        pooler_output = self.pooler_output_layer(sequence_output[0])
        print(f"Final pooler output shape: {pooler_output.shape}")
        return sequence_output, pooler_output

    # 训练模式设置
    def train(self, mode=True):
        self.training = mode
        print(f"Switching to {'train' if mode else 'eval'} mode")

    # 设置优化器
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        print(f"Optimizer set: {type(optimizer).__name__}")

    # 设置损失函数
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn
        print(f"Loss function set: {type(loss_fn).__name__}")

    # 训练步骤 (简化版)
    def train_step(self, x, y):
        if not self.training:
            self.train(True)

        # 前向传播
        seq_out, pool_out = self.forward(x)

        # 计算损失 (简化示例)
        loss = self.loss_fn(pool_out, y) if self.loss_fn else None
        print(f"Loss: {loss}" if loss is not None else "No loss function set")

        # 反向传播 (简化示例 - 实际实现需要计算梯度)
        if self.optimizer:
            print("Performing optimization step...")
            self.optimizer.step()
            self.optimizer.zero_grad()

        return seq_out, pool_out, loss


# 示例使用
if __name__ == "__main__":
    # 加载预训练模型
    print("Loading pre-trained BERT model...")
    bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()

    # 初始化自定义BERT模型
    print("\nInitializing custom BERT model...")
    db = DiyBert(state_dict)

    # 示例输入
    x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子

    # 前向传播
    print("\nPerforming forward pass...")
    diy_sequence_output, diy_pooler_output = db.forward(x)

    # 打印结果
    print("\nResults:")
    print("Sequence output shape:", diy_sequence_output.shape)
    print("Pooler output shape:", diy_pooler_output.shape)
    print("\nSample sequence output (first token):", diy_sequence_output[0])
    print("Pooler output:", diy_pooler_output)

    # 训练示例 (简化版)
    print("\nTraining example (simplified):")


    # 假设我们有一个简单的优化器和损失函数
    class DummyOptimizer:
        def step(self): print("Optimizer step performed")

        def zero_grad(self): print("Gradients zeroed")


    class DummyLoss:
        def __call__(self, pred, true):
            print("Calculating loss...")
            return np.mean((pred - true) ** 2)


    db.set_optimizer(DummyOptimizer())
    db.set_loss_fn(DummyLoss())

    # 模拟训练数据 (随机生成)
    dummy_y = np.random.randn(768)  # 假设我们预测768维向量

    # 执行训练步骤
    db.train_step(x, dummy_y)
