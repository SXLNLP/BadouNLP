# coding:utf8
import random
import json
import torch
import torch.nn as nn
import numpy as np

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法 (此版本默认使用RNN)
"""

class CharPositionModel(nn.Module):
    """
    一个用于字符位置分类的PyTorch模型。
    输入一个字符序列，输出字符'a'在序列中的位置（0到sentence_length-1），
    或者如果'a'不在序列中，则输出一个特殊的类别（sentence_length）。
    """
    def __init__(self, vector_dim, sentence_length, vocab_size):
        """
        初始化模型层。
        Args:
            vector_dim (int): 字符嵌入的维度。
            sentence_length (int): 输入序列的固定长度。
            vocab_size (int): 词汇表的大小。
        """
        super(CharPositionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim)  # 字符嵌入层
        # 使用RNN层来处理序列数据
        # batch_first=True 表示输入数据的维度是 (batch_size, sequence_length, embedding_dim)
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)

        # 分类层：输出维度为 sentence_length + 1
        # +1 是为了处理 'a' 不存在的情况，将其映射到一个额外的类别
        self.classify = nn.Linear(vector_dim, sentence_length + 1)     
        
        # 交叉熵损失函数，适用于多类别分类
        self.loss_fn = nn.functional.cross_entropy

    def forward(self, x, y=None):
        """
        前向传播函数。
        Args:
            x (torch.Tensor): 输入的字符ID序列张量 (batch_size, sentence_length)。
            y (torch.Tensor, optional): 真实的类别标签 (batch_size)。如果提供，则计算损失。
        Returns:
            torch.Tensor: 如果提供了y，返回损失值；否则返回预测的类别得分（logits）。
        """
        # 1. 字符嵌入
        # x_embed 维度: (batch_size, sentence_length, vector_dim)
        x_embed = self.embedding(x)           
        
        # 2. RNN处理
        # rnn_out: 序列中每个时间步的输出 (batch_size, sentence_length, vector_dim)
        # hidden: 最后一个时间步的隐藏状态 (num_layers * num_directions, batch_size, vector_dim)
        # 对于单层RNN，hidden.squeeze(0) 等价于 rnn_out[:, -1, :]
        rnn_out, hidden = self.rnn(x_embed)
        
        # 3. 取RNN最后一个时间步的输出作为特征向量进行分类
        # x_feature 维度: (batch_size, vector_dim)
        x_feature = rnn_out[:, -1, :] 
        
        # 4. 线性分类层
        # y_pred 维度: (batch_size, sentence_length + 1)
        y_pred = self.classify(x_feature)            
        
        # 5. 计算损失或返回预测结果
        if y is not None:
            return self.loss_fn(y_pred, y)   # 预测值和真实值计算损失
        else:
            return y_pred                 # 输出预测结果（logits）

def build_char_vocab():
    """
    构建字符到整数ID的映射（词汇表）。
    Returns:
        dict: 字符到整数ID的映射。
    """
    chars = "abcdefghijk"  # 定义字符集
    vocab = {"pad": 0}     # 'pad' 用于填充，ID为0
    for index, char in enumerate(chars):
        vocab[char] = index + 1 # 其他字符从1开始分配ID
    vocab['unk'] = len(vocab) # 'unk' (未知字符) 放在最后
    return vocab

def generate_single_sample(vocab, sentence_length):
    """
    随机生成一个训练样本。
    Args:
        vocab (dict): 字符词汇表。
        sentence_length (int): 字符串的固定长度。
    Returns:
        tuple: (x, y)
            x (list): 字符序列的整数ID列表。
            y (int): 'a'在序列中的位置，如果不存在则为 sentence_length。
    """
    # 从词汇表中随机选择 sentence_length 个字符
    # 注意：这里使用 random.sample，因此字符不会重复
    # 且要求 vocab.keys() 的长度足够（至少 sentence_length + 1 因为有pad, unk）
    # 为了保证能选择到 'a'，我们将字符集和 'unk' 分开处理，确保能抽到真实字符。
    # 更安全的做法是：
    available_chars = list(set(vocab.keys()) - {'pad', 'unk'})
    if len(available_chars) < sentence_length:
        raise ValueError("Vocabulary is too small for the given sentence_length.")
        
    x_chars = random.sample(available_chars, sentence_length)

    # 确定 'a' 的位置作为标签y
    if "a" in x_chars:
        y = x_chars.index("a")
    else:
        y = sentence_length # 'a' 不存在时，标签为 sentence_length

    # 将字符列表转换为整数ID列表
    x_ids = [vocab.get(char, vocab['unk']) for char in x_chars]
    return x_ids, y

def build_dataset(sample_size, vocab, sentence_length):
    """
    生成一个批次的训练数据集。
    Args:
        sample_size (int): 需要生成的样本数量。
        vocab (dict): 字符词汇表。
        sentence_length (int): 字符串的固定长度。
    Returns:
        tuple: (dataset_x, dataset_y)
            dataset_x (torch.LongTensor): 输入序列的整数ID张量。
            dataset_y (torch.LongTensor): 真实类别标签张量。
    """
    dataset_x = []
    dataset_y = []
    for _ in range(sample_size):
        x, y = generate_single_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, vector_dim, sentence_length):
    """
    创建并返回模型实例。
    Args:
        vocab (dict): 字符词汇表。
        vector_dim (int): 字符嵌入的维度。
        sentence_length (int): 输入序列的固定长度。
    Returns:
        CharPositionModel: 模型实例。
    """
    model = CharPositionModel(vector_dim, sentence_length, len(vocab))
    return model

def evaluate_model(model, vocab, sentence_length, num_test_samples=200):
    """
    评估模型的准确率。
    Args:
        model (nn.Module): 要评估的模型。
        vocab (dict): 字符词汇表。
        sentence_length (int): 输入序列的固定长度。
        num_test_samples (int): 用于测试的样本数量。
    Returns:
        float: 模型的正确率。
    """
    model.eval() # 设置模型为评估模式
    
    # 生成测试数据集
    x_test, y_test = build_dataset(num_test_samples, vocab, sentence_length)
    print(f"本次预测集中共有{len(y_test)}个样本")
    
    correct_predictions = 0
    wrong_predictions = 0
    
    # 在torch.no_grad()上下文中进行预测，不计算梯度以节省内存和计算
    with torch.no_grad():
        y_pred_logits = model(x_test)      # 模型预测，得到logits
        # torch.argmax(dim=1) 获取每个样本预测得分最高的类别ID
        predicted_labels = torch.argmax(y_pred_logits, dim=1) 
        
        # 逐个样本比较预测结果和真实标签
        for y_p, y_t in zip(predicted_labels, y_test):
            if int(y_p) == int(y_t):
                correct_predictions += 1
            else:
                wrong_predictions += 1
    
    accuracy = correct_predictions / (correct_predictions + wrong_predictions)
    print(f"正确预测个数：{correct_predictions}, 错误预测个数：{wrong_predictions}, 正确率：{accuracy:.4f}")
    return accuracy

def train_main():
    """
    主训练函数。
    """
    # 配置参数
    epoch_num = 20        # 训练轮数
    batch_size = 40       # 每次训练的样本个数
    train_sample_total = 1000 # 每轮训练总共使用的样本总数
    vector_dim = 30       # 字符嵌入的维度
    sentence_length = 10  # 样本文本长度 (例如 "abcdefghij")
    learning_rate = 0.001 # 学习率
    

    vocab = build_char_vocab()
    

    model = build_model(vocab, vector_dim, sentence_length)
    print(f"模型结构:\n{model}")
    print(f"词汇表大小: {len(vocab)}")
    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录训练过程中的准确率和损失
    history = []
    
    print("\n========= 开始训练 =========\n")
    for epoch in range(epoch_num):
        model.train() # 设置模型为训练模式
        epoch_losses = [] # 记录当前epoch的损失
        
        # 按照batch_size生成数据并进行训练
        for batch_idx in range(int(train_sample_total / batch_size)):
            # 构造一组训练样本
            x_batch, y_batch = build_dataset(batch_size, vocab, sentence_length) 
            
            optimizer.zero_grad()    # 梯度清零
            loss = model(x_batch, y_batch)   # 计算损失
            loss.backward()          # 反向传播，计算梯度
            optimizer.step()         # 更新模型权重
            
            epoch_losses.append(loss.item())
            
        avg_loss = np.mean(epoch_losses)
        print(f"=========\n第{epoch + 1}轮平均Loss: {avg_loss:.4f}")
        
        # 每轮训练结束后，评估模型性能
        acc = evaluate_model(model, vocab, sentence_length)   
        history.append([acc, avg_loss])

    print("\n========= 训练结束 =========\n")

    model_save_path = "char_position_rnn_model.pth"
    vocab_save_path = "char_position_vocab.json"
    
    torch.save(model.state_dict(), model_save_path)
    print(f"模型权重已保存至: {model_save_path}")
    
    with open(vocab_save_path, "w", encoding="utf8") as writer:
        json.dump(vocab, writer, ensure_ascii=False, indent=2)
    print(f"词汇表已保存至: {vocab_save_path}")
    
    return model_save_path, vocab_save_path

def predict_with_model(model_path, vocab_path, input_strings):


    vector_dim = 30  
    sentence_length = 10 

    with open(vocab_path, "r", encoding="utf8") as reader:
        vocab = json.load(reader)
    print(f"加载词汇表完成，大小: {len(vocab)}")

    model = build_model(vocab, vector_dim, sentence_length)     
    model.load_state_dict(torch.load(model_path))             
    model.eval()   # 设置模型为评估模式

    input_ids_list = []
    for input_string in input_strings:
        # 将输入字符串转换为字符ID序列
        # 使用 .get() 并指定 'unk' 处理未知字符
        # 同时确保输入字符串长度与 sentence_length 匹配，这里假设是匹配的
        if len(input_string) != sentence_length:
            print(f"警告: 输入字符串 '{input_string}' 长度不匹配 sentence_length ({sentence_length})。可能导致错误预测。")
        char_ids = [vocab.get(char, vocab['unk']) for char in input_string]
        input_ids_list.append(char_ids)
    

    input_tensor = torch.LongTensor(input_ids_list)


    with torch.no_grad():  # 不计算梯度
        result_logits = model.forward(input_tensor)  # 模型预测，得到logits
    

    print("\n========= 预测结果 =========\n")
    for i, input_string in enumerate(input_strings):

        predicted_label = torch.argmax(result_logits[i]).item()
        

        probabilities = torch.softmax(result_logits[i], dim=0)
        

        prob_str = ", ".join([f"{p:.3f}" for p in probabilities.tolist()])
        

        predicted_position_info = f"位置 {predicted_label}" if predicted_label < sentence_length else "不存在"
        
        print(f"输入: '{input_string}'")
        print(f"  预测 'a' 的位置: {predicted_position_info}")
        print(f"  各位置概率分布: [{prob_str}]")
        print("-" * 30)


if __name__ == "__main__":

    model_path, vocab_path = train_main()

    test_strings = [
        "kijabcdefh",  # 'a' 在位置 3
        "gijkbcdeaf",  # 'a' 在位置 8
        "gkijadfbec",  # 'a' 在位置 5
        "kijhdefacb",  # 'a' 在位置 7
        "bcdefghijk",  # 'a' 不存在
        "jihgfedcbk",  # 'a' 不存在
        "abcdefghij"   # 'a' 在位置 0
    ]
    predict_with_model(model_path, vocab_path, test_strings)
