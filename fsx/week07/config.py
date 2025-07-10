Config = {

    # 路径配置
    "model_path": "model",
    "train_data_path": "../data2/train.csv",
    "valid_data_path": "../data2/valid.csv",
    "vocab_path": "../data2/vocab/chars.txt",
    "pretrain_model_path": r"/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese",

    # 模型配置
    "model_type": "bert",
    "num_layers": 2,  # 网络层数（如LSTM/BERT层数
    "hidden_size": 256,
    "kernel_size": 3,  # 卷积核大小
    "dropout": 0.5,
    "max_length": 30,
    "class_num": 0,  # 分类数 会跟随数据变化而变化 在数据加载中设置
    "vocab_size": 0,  # 词表大小 会跟随数据变化而变化 在数据加载中设置

    # 训练配置
    "epoch": 3,
    "batch_size": 128,
    "learning_rate": 1e-5,
    "optimizer": "adam",
    "pooling_style": "mean",
    "seed": 10086,  # 随机种子：固定随机数生成器，保证结果可复现

    # 其他配置
    "train_ratio": 0.8,
    "gpu_switch": True,
    "is_csv": True,
    "is_just_train_data": True
}
