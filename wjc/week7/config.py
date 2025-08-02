# 修改数据路径和分类参数
Config = {
    "data_path": r"D:\BaiduNetdiskDownload\badouai\recored\第七周\pretrain.csv",  # 新数据集路径
    "model_path": "output",
    "train_data_path": r"D:\BaiduNetdiskDownload\badouai\recored\第七周\pretrain.csv",
    "valid_data_path": r"D:\BaiduNetdiskDownload\badouai\recored\第七周\pretrainnew.csv",
    "vocab_path": "chars.txt",
    "model_type": "lstm",
    "max_length": 50,  # 增加最大长度适应评论
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 1,  # 减少epoch数
    "batch_size": 64,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\BaiduNetdiskDownload\badouai\recored\bert-base-chinese",
    "seed": 987,
    "class_num": 2,  # 二分类
    "enviroment": "test"
}
