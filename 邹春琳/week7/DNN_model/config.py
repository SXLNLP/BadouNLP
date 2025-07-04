"""配置参数信息"""

Config = {
    "result_compare_path": "output",
    "data_path": r"e-commerce_comments.csv",
    "test_size": 0.2,
    "vocab_path": "chars.txt",
    "model_path": "save_model",
    "model_type": "rnn",
    "max_length": 30,
    "class_num": 2,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 128,
    "pooling_style": "abg",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "seed": 987
}