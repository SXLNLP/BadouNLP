"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "data/train.csv",
    "valid_data_path": "data/valid.csv",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 100,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":r"D:\models\google-bert\bert-base-chinese",
    "seed": 987
}
