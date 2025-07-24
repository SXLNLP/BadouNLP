
# from transformers import BertModel

# bert = BertModel.from_pretrained(r"/home/nbs07/lean1/bert-base-chinese", return_dict=False)
# state_dict = bert.state_dict()
# total_params = 0
# for p in state_dict.values():
#     total_params += p.numel()
# for name, param in state_dict.items():
#     print(f"{name}: {param.shape} - {param.numel()}")  # 打印每个参数的名称、形状和元素数量

# 手动计算bert模型总参数量
def count_parameters():
    total_params = 0
    vocab_size = 21128  # bert-base-chinese的词表大小
    hidden_size = 768  # bert-base-chinese的输出维度
    intermediate_size = 4 * hidden_size  # feedforward层的中间层大小
    # embedding层
    embeddings_token_embedding = vocab_size * hidden_size
    embeddings_segment_embedding = 2 * hidden_size  # segment embedding有两个类别
    embeddings_position_embedding = 512 * hidden_size
    # embedding-归一化层
    embeddings_layer_norm_weight = hidden_size
    embeddings_layer_norm_bias = hidden_size
    total_params += (embeddings_token_embedding + embeddings_segment_embedding +
                     embeddings_position_embedding + embeddings_layer_norm_weight +
                     embeddings_layer_norm_bias)
    print("embedding层参数量：", total_params)
    # transformer层
    transformer_params = 0
    num_layers = 12
    QW= hidden_size * hidden_size  # QueryW矩阵
    QB = hidden_size
    KW = hidden_size * hidden_size  # KeyW矩阵
    KB = hidden_size
    VW = hidden_size * hidden_size  # ValueW矩阵
    VB = hidden_size 
    # attention输出
    attention_output_W = hidden_size * hidden_size
    attention_output_B = hidden_size
    # attention输出归一化
    attention_output_layer_norm_weight = hidden_size
    attention_output_layer_norm_bias = hidden_size
    print("单层attention参数量：", QW + QB + KW + KB + VW + VB +
          attention_output_W + attention_output_B +
          attention_output_layer_norm_weight + attention_output_layer_norm_bias)
    # feedforward层
    feedforward_intermediate_W = hidden_size * intermediate_size
    feedforward_intermediate_B = intermediate_size
    feedforward_output_W = intermediate_size * hidden_size
    feedforward_output_B = hidden_size
    # feedforward归一化
    feedforward_layer_norm_weight = hidden_size
    feedforward_layer_norm_bias = hidden_size
    print("单层feedforward参数量：", feedforward_intermediate_W + feedforward_intermediate_B +
          feedforward_output_W + feedforward_output_B +
          feedforward_layer_norm_weight + feedforward_layer_norm_bias)
    transformer_params += (QW + QB + KW + KB + VW + VB +
                           attention_output_W + attention_output_B +
                           attention_output_layer_norm_weight +
                           attention_output_layer_norm_bias +
                           feedforward_intermediate_W + feedforward_intermediate_B +
                           feedforward_output_W + feedforward_output_B +
                           feedforward_layer_norm_weight + feedforward_layer_norm_bias)
    # 每层transformer的参数量
    print("单层transformer参数量：", transformer_params)
    transformer_params = transformer_params * num_layers  # 12层transformer的参数量
    print("12层transformer参数量：", transformer_params)
    total_params += transformer_params
    # 输出-线性层
    output_W = hidden_size * hidden_size
    output_B = hidden_size
    total_params += (output_W + output_B)
    return total_params

def main():
    total_params = count_parameters()
    print("总参数量：", total_params)
    # print("bert模型参数量：", sum(p.numel() for p in bert.parameters()))  # 使用transformers库计算的参数量

if __name__ == "__main__":
    main()

