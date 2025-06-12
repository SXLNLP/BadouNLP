import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow_addons.layers import CRF
from tensorflow.keras.preprocessing.sequence import pad_sequences
from itertools import groupby

def generate_all_cut_labels(s):
    """生成字符串所有可能的切分标签序列"""
    n = len(s)
    if n == 0:
        return []
    
    # 所有可能的切分点组合（每个字符后面都可以选择切或不切）
    cut_points = []
    for i in range(1, n):
        # 生成所有可能的切分点组合（0表示不切，1表示切）
        # 注意：最后一个字符后面不能切分
        for combo in product([0, 1], repeat=n-1):
            # 跳过连续切分的情况（会产生空子串）
            if any(combo[j] == 1 and combo[j+1] == 1 for j in range(len(combo)-1)):
                continue
            cut_points.append(combo)
    
    # 如果没有切分点（单字符情况），添加完整字符串标签
    if not cut_points:
        return [[0] * n]
    
    # 将切分点组合转换为标签序列
    all_labels = []
    for cuts in cut_points:
        labels = []
        # 第一个字符总是B
        labels.append(0)  # B
        for i in range(1, n):
            # 如果前一个位置有切分，当前位置是B
            if cuts[i-1] == 1:
                labels.append(0)  # B
            # 否则是I
            else:
                labels.append(1)  # I
        all_labels.append(labels)
    
    return all_labels

def product(iterable, repeat=1):
    """实现itertools.product的功能"""
    pools = [tuple(iterable)] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

def create_dataset(strings, max_len=20):
    """创建训练数据集"""
    char2idx = {}
    idx2char = {}
    
    # 构建字符词典
    chars = set()
    for s in strings:
        chars.update(s)
    chars = sorted(chars)
    char2idx = {c: i+1 for i, c in enumerate(chars)}  # 0用于padding
    idx2char = {i+1: c for i, c in enumerate(chars)}
    
    # 生成样本和标签
    X, y = [], []
    for s in strings:
        # 生成所有可能的切分标签
        all_labels = generate_all_cut_labels(s)
        
        # 为每个切分方式创建样本
        for labels in all_labels:
            # 将字符转换为索引
            char_indices = [char2idx[c] for c in s]
            X.append(char_indices)
            y.append(labels)
    
    # 填充序列
    X = pad_sequences(X, maxlen=max_len, padding='post', value=0)
    y = pad_sequences(y, maxlen=max_len, padding='post', value=-1)  # -1表示填充位置
    
    return np.array(X), np.array(y), char2idx, idx2char

def build_model(vocab_size, max_len, num_tags=2):
    """构建BiLSTM-CRF模型"""
    # 输入层
    inputs = layers.Input(shape=(max_len,))
    
    # 嵌入层
    embedding = layers.Embedding(
        input_dim=vocab_size + 1,  # +1 for padding
        output_dim=64,
        mask_zero=True
    )(inputs)
    
    # BiLSTM层
    bilstm = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True)
    )(embedding)
    
    # 全连接层
    dense = layers.TimeDistributed(
        layers.Dense(num_tags)
    )(bilstm)
    
    # CRF层
    crf = CRF(num_tags)
    outputs = crf(dense)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=crf.loss,
        metrics=[crf.accuracy]
    )
    
    return model

def decode_cuts(s, labels):
    """根据标签序列解码切分结果"""
    # 移除填充和无效标签
    valid_labels = labels[:len(s)]
    
    # 根据标签分组
    groups = []
    for k, g in groupby(enumerate(valid_labels), key=lambda x: x[1]):
        if k == -1:  # 跳过填充
            continue
        indices = [i for i, _ in g]
        groups.append(indices)
    
    # 构建切分结果
    cuts = []
    start = 0
    for group in groups:
        end = group[-1] + 1
        cuts.append(s[start:end])
        start = end
    
    return cuts

def predict_all_cuts(model, s, char2idx, max_len=20):
    """预测字符串的所有可能切分"""
    # 转换为模型输入
    char_indices = [char2idx.get(c, 0) for c in s]
    padded = pad_sequences([char_indices], maxlen=max_len, padding='post', value=0)
    
    # 预测标签概率
    probs = model.predict(padded)[0]
    
    # 生成所有可能的标签组合
    n = len(s)
    all_labels = generate_all_cut_labels(s)
    
    # 计算每种切分方式的概率
    results = []
    for labels in all_labels:
        # 扩展标签序列到最大长度
        padded_labels = list(labels) + [-1] * (max_len - n)
        
        # 计算路径概率
        log_prob = 0
        for i, tag in enumerate(padded_labels[:max_len]):
            if tag == -1:  # 跳过填充
                continue
            log_prob += probs[i, tag]
        
        # 解码切分结果
        cuts = decode_cuts(s, labels)
        results.append((cuts, np.exp(log_prob)))
    
    # 按概率排序
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# 示例使用
if __name__ == "__main__":
    # 训练数据
    train_strings = ["abc", "def", "hello", "world", "test", "deep", "learning"]
    
    # 创建数据集
    max_len = 10
    X, y, char2idx, idx2char = create_dataset(train_strings, max_len)
    vocab_size = len(char2idx)
    
    # 构建模型
    model = build_model(vocab_size, max_len)
    
    # 训练模型（实际应用中需要更多数据和迭代）
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
    
    # 测试字符串
    test_string = "abc"
    
    # 预测所有可能的切分
    all_cuts = predict_all_cuts(model, test_string, char2idx, max_len)
    
    # 输出结果
    print(f"字符串 '{test_string}' 的所有可能切分:")
    for i, (cuts, prob) in enumerate(all_cuts):
        print(f"{i+1}. {cuts} (概率: {prob:.4f})")
