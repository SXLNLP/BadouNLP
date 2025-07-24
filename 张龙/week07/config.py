import os
import torch  # 导入 torch 库

class Config:
    def __init__(self):
        # Model Configuration
        self.model_type = 'bert'  # 选择模型类型：'bert', 'lstm', 或 'rnn'
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.num_epochs = 1
        self.max_seq_length = 64
        self.random_seed = 42  # 随机种子，确保结果可复现
        self.test_size = 0.2  # 测试集占比
        self.dropout_rate = 0.3  # Dropout 的比率，防止过拟合

        # Path configurations
        self.save_model_dir = './saved_models'  # 模型保存路径
        self.saved_model_path = './saved_models/text_classification_model.pt'  # 加载训练好的模型路径
        self.load_pretrained_model = False  # 是否加载已训练好的模型，True 加载，False 从头训练
        self.data_path = r'D:\My Documents\11182971\Documents\xwechat_files\wxid_wjsnouyqo1eh22_95f8\msg\file\2025-06\week7 文本分类问题\week7 文本分类问题\文本分类练习.csv'  # 数据文件的路径（你需要修改为实际的路径）

        # Device Configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

