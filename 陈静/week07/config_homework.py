Config = {
    "seed": 987,
    "pretrain_model_path": r"C:\Users\cj783\Desktop\AI算法工程师\week6\week6 语言模型和预训练\bert-base-chinese",

    # 模型输入与规模
    "max_length": 32,           
    "batch_size": 16,           

    # 模型学习率与训练参数
    "learning_rate": 3e-5,     
    "hidden_size": 256,         
    "epoch": 1,                 

    # 保留字段
    "model_path": "output",
    "vocab_path": "chars.txt",
    "model_type": "bert",
    "kernel_size": 3,
    "num_layers": 2,
    "pooling_style": "max",
    "optimizer": "adam",
}
