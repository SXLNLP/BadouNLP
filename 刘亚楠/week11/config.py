import os


class Config:
    # 数据配置
    data_path = "/Users/nlp/week10/week10 文本生成问题/week10 文本生成问题/transformers-生成文章标题/sample_data.json"
    max_length = 128
    bos_token = "[BOS]"
    eos_token = "[EOS]"

    # 模型配置
    pretrained_model = "/Users/nlp/week9/p_week9/p_ner/bert-base-chinese"
    batch_size = 8
    learning_rate = 3e-5
    num_epochs = 10

    # 训练配置
    output_dir = "output"
    save_steps = 500
    logging_steps = 100


config = Config()