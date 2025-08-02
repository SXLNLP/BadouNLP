# -*- coding: utf-8 -*-
Config = {
    "model_path": r"C:\Users\cj783\Desktop\AI算法工程师\week9\第九周 序列标注(1)\ner\homework\model_output",
    "schema_path": r"C:\Users\cj783\Desktop\AI算法工程师\week9\第九周 序列标注(1)\ner\ner_data\schema.json",
    "train_data_path": r"C:\Users\cj783\Desktop\AI算法工程师\week9\第九周 序列标注(1)\ner\ner_data\train",
    "valid_data_path": r"C:\Users\cj783\Desktop\AI算法工程师\week9\第九周 序列标注(1)\ner\ner_data\test",
    "vocab_path": r"C:\Users\cj783\Desktop\AI算法工程师\week9\第九周 序列标注(1)\ner\chars.txt",
    "max_length": 100,
    "hidden_size": 768,  # 改为BERT的隐藏层大小
    "num_layers": 12,    # 改为BERT的层数
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT需要更小的学习率
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"C:\Users\cj783\Desktop\AI算法工程师\week6\week6 语言模型和预训练\bert-base-chinese"
}
