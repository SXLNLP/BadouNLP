Config = {
    "model_path": "model_output",
    "model_type": "bert_mid_layer",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "tuning_tactics":"lora_tuning",
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"bert-base-chinese"
}
