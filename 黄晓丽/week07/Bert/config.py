# config.py
import torch
from transformers import BertTokenizer, BertConfig


class Config:
    # 数据路径
    data_path = r"D:\BaiduNetdiskDownload\aaabadouDownload\文本分类练习.csv"

    # BERT 模型配置
    pretrained_model_name = 'bert-base-chinese'  # 使用中文BERT
    max_sequence_length = 128  # BERT最大序列长度
    num_classes = 2  # 分类类别数（好评/差评）

    # 训练参数
    batch_size = 16  # BERT通常使用较小的batch size
    learning_rate = 2e-5  # BERT推荐的学习率
    num_epochs = 3  # BERT通常训练3-4个epoch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 保存路径
    model_save_path = 'best_bert_model.bin'
    log_file = 'training_log.csv'

    # 列名配置
    text_column = 'review'
    label_column = 'label'

    # 性能测试参数
    predict_test_size = 1000

    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    def __str__(self):
        """返回配置的字符串表示"""
        return (
            f"BERT 配置:\n"
            f"  预训练模型: {self.pretrained_model_name}\n"
            f"  最大序列长度: {self.max_sequence_length}\n"
            f"  学习率: {self.learning_rate}\n"
            f"  批量大小: {self.batch_size}\n"
            f"  训练轮次: {self.num_epochs}\n"
            f"  设备: {self.device}\n"
        )


# 创建全局配置实例
global_config = Config()
