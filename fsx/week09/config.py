Config = {
    "model_path": "model_output",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "predict_data_path": "ner_data/dev",
    "vocab_path": "chars.txt",
    "schema_path": "ner_data/schema.json",
    "bert_vocab_path": "/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese/vocab.txt",

    "batch_size": 32,  # 每次数据的批次大小 会影响训练速度
    "epochs": 10,  # 训练的轮数
    "max_length": 30,  # 应该是训练数据长度的p90/99 先随便写
    "hidden_size": 128,
    "vocab_size": 0,  # 词表大小 在加载数据的时候赋值
    "optimizer": "adam",  # 优化器
    "learning_rate": 1e-3,  # 学习率
    "class_num": 9,  # 类别数 和 schema 有关
    "num_layers": 2,  # rnn 的层数
    "pooling_style": "avg",
    "seed": 10086,

    "bert_path": r"/Users/juewang/Downloads/八斗/第六周/week6 语言模型和预训练/bert-base-chinese",

    "use_bert_switch": True,
    "use_crf": False

}
