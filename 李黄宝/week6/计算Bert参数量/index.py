import torch
import math
import numpy as np
from transformers import BertModel

# 加载预训练模型
bert = BertModel.from_pretrained(r"/Users/chenayu/Desktop/111/bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()

print("=" * 80)
print("Bert模型参数量计算")
print("=" * 80)

# 计算总参数量
total_params = sum(p.numel() for p in bert.parameters())
print(f"\nBert模型总参数量: {total_params} (约{total_params/1e6:.1f}百万)")

# 计算可训练参数量
trainable_params = sum(p.numel() for p in bert.parameters() if p.requires_grad)
print(f"可训练参数量: {trainable_params}")

# 按层类型分类统计
param_counts = {
    "embeddings": 0,
    "attention": 0,
    "intermediate": 0,
    "output": 0,
    "pooler": 0,
    "layer_norm": 0
}

for name, param in state_dict.items():
    num_params = param.numel()
    if "embeddings" in name:
        param_counts["embeddings"] += num_params
    elif "attention" in name:
        param_counts["attention"] += num_params
    elif "intermediate" in name:
        param_counts["intermediate"] += num_params
    elif "output" in name and "dense" in name:  # 排除output.LayerNorm
        param_counts["output"] += num_params
    elif "pooler" in name:
        param_counts["pooler"] += num_params
    elif "LayerNorm" in name:  # 所有的LayerNorm参数
        param_counts["layer_norm"] += num_params

# 打印分类统计结果
print("\n参数分类统计:")
print("-" * 50)
print(f"嵌入层 (Embeddings): {param_counts['embeddings']} (占比: {param_counts['embeddings']/total_params*100:.1f}%)")
print(f"注意力层 (Attention): {param_counts['attention']} (占比: {param_counts['attention']/total_params*100:.1f}%)")
print(f"中间层 (Intermediate): {param_counts['intermediate']} (占比: {param_counts['intermediate']/total_params*100:.1f}%)")
print(f"输出层 (Output): {param_counts['output']} (占比: {param_counts['output']/total_params*100:.1f}%)")
print(f"池化层 (Pooler): {param_counts['pooler']} (占比: {param_counts['pooler']/total_params*100:.1f}%)")
print(f"归一化层 (LayerNorm): {param_counts['layer_norm']} (占比: {param_counts['layer_norm']/total_params*100:.1f}%)")

# 验证总和
calculated_total = sum(param_counts.values())
print("\n验证统计:")
print(f"分类统计总和: {calculated_total}")
print(f"与总参数量匹配: {calculated_total == total_params}")

print("\n" + "=" * 80)
print("Bert模型输出验证")
print("=" * 80)

# 原始测试代码
x = np.array([2450, 15486, 102, 2110])   # 假想成4个字的句子
torch_x = torch.LongTensor([x])          # pytorch形式输入
seqence_output, pooler_output = bert(torch_x)
print("\n原始Bert输出形状:")
print(seqence_output.shape, pooler_output.shape)

# 打印所有权重名称
print("\nBert权重键名:")
print(bert.state_dict().keys())

# 辅助函数
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))
    # return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3)))

class DiyBert:
    # 将预训练好的整个权重字典输入进来
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 12  # 修改为12层以匹配bert-base-chinese
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        # transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        # pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()


    # bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [max_len, hidden_size]
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    # embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])

    # 执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    # 执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        # 取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        # self attention层
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        # bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        # feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        # bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
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
        # x.shape = max_len * hidden_size
        # q_w, k_w, v_w  shape = hidden_size * hidden_size
        # q_b, k_b, v_b  shape = hidden_size
        q = np.dot(x, q_w.T) + q_b  # shape: [max_len, hidden_size]      W * X + B lINER
        k = np.dot(x, k_w.T) + k_b  # shpae: [max_len, hidden_size]
        v = np.dot(x, v_w.T) + v_b  # shpae: [max_len, hidden_size]
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))  
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    # 多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    # 前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    # 归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    # 链接[cls] token的输出层
    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    # 最终输出
    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output


print("\n" + "=" * 80)
print("自定义DiyBert实现验证")
print("=" * 80)

# 自制模型
db = DiyBert(state_dict)
diy_sequence_output, diy_pooler_output = db.forward(x)

# torch模型
torch_sequence_output, torch_pooler_output = bert(torch_x)

# 比较输出
print("\n序列输出比较 (前5个元素):")
print("自定义实现:", diy_sequence_output[0][0][:5])
print("原始Bert输出:", torch_sequence_output[0][0].detach().numpy()[:5])

print("\n池化输出比较:")
print("自定义实现:", diy_pooler_output[:5])
print("原始Bert输出:", torch_pooler_output[0].detach().numpy()[:5])

# 计算差异
seq_diff = np.mean(np.abs(diy_sequence_output - torch_sequence_output.detach().numpy()))
pool_diff = np.mean(np.abs(diy_pooler_output - torch_pooler_output[0].detach().numpy()))
print(f"\n输出差异评估:")
print(f"序列输出平均绝对差异: {seq_diff:.6f}")
print(f"池化输出平均绝对差异: {pool_diff:.6f}")

print("\n" + "=" * 80)
print("程序执行完毕")
print("=" * 80)
