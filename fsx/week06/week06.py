import torch
import torch.nn as nn
from transformers import BertModel

"""
计算bert模型参数
"""

def computer_parameters(model: BertModel):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def computer_layer_parameters(model: BertModel):
    layer_params = {}

    embedding_params = sum(p.numel() for p in model.embeddings.parameters())
    layer_params["Embedding"] = embedding_params

    pooler_params = sum(p.numel() for p in model.pooler.parameters())
    layer_params["Pooler"] = pooler_params

    encoder_params = 0
    for i, layer in enumerate(model.encoder.layer):
        layer_name = f"Encoder Layer {i + 1}"
        layer_p = sum(p.numel() for p in layer.parameters())
        layer_params[layer_name] = layer_p
        encoder_params += layer_p

    return layer_params, embedding_params, encoder_params, pooler_params


def main():
    # 加载预训练的BERT模型
    cache_dir = "bert-base-chinese"
    model = BertModel.from_pretrained("bert-base-chinese", cache_dir=cache_dir)

    total_params, trainable_params = computer_parameters(model)

    layer_params, embedding_params, encoder_params, pooler_params = computer_layer_parameters(model)

    # 打印结果
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print("\n各层参数分布:")

    # 打印嵌入层参数
    print(f"  - 嵌入层: {embedding_params:,} ({embedding_params / total_params * 100:.2f}%)")

    # 打印编码器层参数
    for i in range(len(model.encoder.layer)):
        layer_name = f"Encoder Layer {i + 1}"
        print(f"  - {layer_name}: {layer_params[layer_name]:,} ({layer_params[layer_name] / total_params * 100:.2f}%)")

    # 打印池化层参数
    print(f"  - 池化层: {pooler_params:,} ({pooler_params / total_params * 100:.2f}%)")

    # 打印参数比例
    print("\n参数比例:")
    print(f"  - 嵌入层占比: {embedding_params / total_params * 100:.2f}%")
    print(f"  - 编码器层占比: {encoder_params / total_params * 100:.2f}%")
    print(f"  - 池化层占比: {pooler_params / total_params * 100:.2f}%")


if __name__ == '__main__':
    main()
