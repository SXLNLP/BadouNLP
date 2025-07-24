# 配置文件：存储模型和训练的参数
import os

class Config:
    # 数据路径（请修改为你的实际路径）
    data_path = r"E:\BaiduNetdiskDownload\第七周 文本分类\week7 文本分类问题\week7 文本分类问题\文本分类练习.csv"
    save_dir = "saved_models"  # 模型保存的文件夹

    # 数据处理参数
    max_seq_length = 128  # 文本最大长度
    test_size = 0.2  # 验证集比例（20%的数据用于验证）
    batch_size = 32

    # 模型参数
    embedding_dim = 100  # 词向量的维度
    vocab_size = 5000  # 词表大小

    # 训练参数
    num_epochs = 5  # 训练轮数
    learning_rate = 0.001  # 学习率

    # 要训练的模型类型
    models = ["rnn", "lstm", "textcnn", "gated_cnn"]
    seed = 42  # 设置随机种子

# 创建配置实例，方便其他文件使用
config = Config()
# 确保保存模型的文件夹存在
os.makedirs(config.save_dir, exist_ok=True)
