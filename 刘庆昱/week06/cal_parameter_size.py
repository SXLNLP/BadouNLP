def cal_embedding_parameter_quantity(dict_len, hidden_dimensions, position_nums, segment_nums):
    embedding_total = 0
    embedding_total += dict_len * hidden_dimensions
    embedding_total += position_nums * hidden_dimensions
    embedding_total += segment_nums * hidden_dimensions
    return embedding_total


def cal_transformer_parameter_quantity(hidden_dimensions):
    transformer_total = 0

    # self_attention 层的参数量
    # K,Q,V 三个变换矩阵的参数量
    self_attention_total = 0
    self_attention_total += 3 * (hidden_dimensions * hidden_dimensions + hidden_dimensions)
    # 线性变换参数量
    self_attention_total += hidden_dimensions * hidden_dimensions + hidden_dimensions

    transformer_total += self_attention_total

    # 层归一化参数量
    layer_norm_total1 = 0
    layer_norm_total1 += 2 * hidden_dimensions

    transformer_total += layer_norm_total1

    # 正向反馈层
    forward_total = 0
    forward_total += (4 * hidden_dimensions) * hidden_dimensions + (4 * hidden_dimensions)
    forward_total += hidden_dimensions * (4 * hidden_dimensions) + hidden_dimensions

    transformer_total += forward_total

    # 层归一化参数量
    layer_norm_total2 = 0
    layer_norm_total2 += 2 * hidden_dimensions

    transformer_total += layer_norm_total2

    return transformer_total


def cal_parameter_quantity(dict_len, hidden_dimensions, position_nums, segment_nums, transformer_layer_num):
    total = 0
    # embedding 层参数量
    total += cal_embedding_parameter_quantity(dict_len, hidden_dimensions, position_nums, segment_nums)
    # transformer 层参数量
    total += cal_transformer_parameter_quantity(hidden_dimensions) * transformer_layer_num
    # pooling 层参数量
    total += hidden_dimensions * hidden_dimensions + hidden_dimensions

    return total


def main():
    dict_len = 21128
    hidden_dimensions = 768
    position_nums = 512
    segment_nums = 2
    transformer_layer_num = 12
    parameter_count = cal_parameter_quantity(dict_len, hidden_dimensions, position_nums, segment_nums, transformer_layer_num)

    # 每个参数占4字节 (32位 = 4字节)
    total_bytes = parameter_count * 4

    # 转换为MB (1 MB = 1024 * 1024 字节)
    total_mb = total_bytes / (1024 * 1024)

    print(f"Bert模型的参数总量: {parameter_count:,} 个")
    print(f"以32位浮点数存储时的大小: {total_mb:.2f} MB")

    # 16位浮点数 (半精度)
    total_mb_16bit = total_bytes / (1024 * 1024) * 0.5
    print(f"以16位浮点数存储时的大小: {total_mb_16bit:.2f} MB")

    # 8位整型 (量化)
    total_mb_8bit = total_bytes / (1024 * 1024) * 0.25
    print(f"以8位整型存储时的大小: {total_mb_8bit:.2f} MB")
    return


if __name__ == "__main__":
    main()
