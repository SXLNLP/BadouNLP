# -*- coding: utf-8 -*-
import os
"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": os.path.dirname(os.path.abspath(__file__))+"/ner_data/schema.json",
    "train_data_path": os.path.dirname(os.path.abspath(__file__))+"/ner_data/train",
    "valid_data_path": os.path.dirname(os.path.abspath(__file__))+"/ner_data/test",
    "vocab_path":os.path.dirname(os.path.abspath(__file__))+"/chars.txt",
    "max_length": 100,
    "hidden_size": 768, #这里一定要与bert的隐藏层一致，如果不一致，以bert为准，因为修改bert就需要重新训练，没法用了
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "model":"bert",#["bert","LSTM"]
    "bert_tokenizer":"custom", #["origin","custom"]
    "use_crf": False,
    "class_num": 9,
    "bert_path": os.path.dirname(os.path.abspath(__file__))+"/../../models/bert-base-chinese"
}

