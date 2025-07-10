# config.py
import time


class Config:
    # 数据路径
    data_path = r"D:\BaiduNetdiskDownload\aaabadouDownload\文本分类练习.csv"

    # 模型选择
    model_type = 'FastText'  # 可选: 'TextCNN' 或 'FastText'

    # FastText 参数 (Gensim 兼容)
    vector_size = 128  # 词向量维度
    window = 5  # 上下文窗口大小
    min_count = 2  # 最小词频
    epochs = 10  # 训练轮次
    workers = 4  # 并行工作数
    sg = 1  # 训练算法: 0=CBOW, 1=Skip-gram
    hs = 0  # 分层softmax: 0=负采样, 1=分层softmax
    negative = 5  # 负采样数量

    # 分类器参数
    classifier_epochs = 100  # 分类器训练轮次
    classifier_lr = 0.001  # 分类器学习率

    # 训练参数
    test_size = 0.2  # 验证集比例

    # 保存路径
    vector_model_path = 'fasttext_vectors.model'  # 词向量模型
    classifier_model_path = 'fasttext_classifier.model'  # 分类器模型
    log_file = 'training_log.csv'

    # 列名配置
    text_column = 'review'
    label_column = 'label'

    # 性能测试参数
    predict_test_size = 1000

    def __str__(self):
        """返回配置的字符串表示"""
        return (
            f"FastText (Gensim) 配置:\n"
            f"  词向量维度: {self.vector_size}\n"
            f"  上下文窗口: {self.window}\n"
            f"  最小词频: {self.min_count}\n"
            f"  训练轮次: {self.epochs}\n"
            f"  训练算法: {'Skip-gram' if self.sg == 1 else 'CBOW'}\n"
            f"  负采样: {self.negative}\n"
            f"  分类器训练轮次: {self.classifier_epochs}\n"
            f"  分类器学习率: {self.classifier_lr}\n"
        )


# 创建全局配置实例
global_config = Config()
