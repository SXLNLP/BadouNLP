
from transformers import BertModel


def bertparams():
    # 加载模型
    model = BertModel.from_pretrained(r"/Volumes/niuniu/aitech/model/bert-base-chinese")

    # 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params} (约 {total_params / 1e6:.2f}M)")

    # 可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params}")

    # 按层统计
    print("\n各层参数量:")
    for name, param in model.named_parameters():
        print(f"{name:<60} {param.numel()}")

if __name__ == "__main__":
    bertparams()
