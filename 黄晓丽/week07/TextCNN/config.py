# config.py - 模型配置参数

import torch

class Config:
    # 数据路径
    data_path = r"D:\BaiduNetdiskDownload\aaabadouDownload\文本分类练习.csv"

    text_column = 'review'
    label_column = 'label'
    # 文本处理
    max_sequence_length = 100  # 最大序列长度
    min_word_freq = 2  # 最小词频

    # 模型参数
    vocab_size = 10000  # 设置默认值（运行时会被覆盖）
    embed_dim = 128  # 词向量维度（隐藏大小）
    num_filters = 100  # 每种卷积核的数量
    filter_sizes = [2, 3, 4]  # 不同尺寸的卷积核
    dropout = 0.5  # Dropout概率
    num_classes = 1  # 输出类别数

    # 训练参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备

    # 保存路径
    model_save_path = 'best_textcnn_model.pth'
    log_file = 'training_log.csv'

    # 性能测试参数
    predict_test_size = 1000  # 预测性能测试的样本量

    def __str__(self):
        """返回配置的字符串表示"""
        return (
            f"TextCNN 配置:\n"
            f"  词向量维度 (hidden_size): {self.embed_dim}\n"
            f"  学习率 (learning_rate): {self.learning_rate}\n"
            f"  卷积核: {self.filter_sizes} x {self.num_filters}\n"
            f"  Dropout: {self.dropout}\n"
            f"  批量大小: {self.batch_size}\n"
            f"  最大序列长度: {self.max_sequence_length}\n"
            f"  设备: {self.device}\n"
            f"  训练轮次: {self.num_epochs}\n"
        )

# 创建全局配置实例
global_config = Config()
