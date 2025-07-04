import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import jieba
from gensim.models import Word2Vec

import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data)
        self.label = torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# 加载语料csv数据
def load_csv_data(filename):
    df = pd.read_csv(filename, sep=',', names=None, index_col=None, usecols=None, dtype=None)
    df_1 = df[df['label'] == 1]
    df_0 = df[df['label'] == 0]
    print(f'负样本有{df_0.shape[0]}条数据, 正样本有{df_1.shape[0]}条数据')
    return df

# 加载停用词
def load_stop_words(path):
    stop_words = []
    with open(path, 'r', encoding='utf-8') as f:
        con = f.readlines()
        for line in con:
            stop_words.append(line.replace('\n', ''))
    return stop_words

# 构造词表
def build_vocab(x_cut_words):
    vocab = []
    for words in x_cut_words:
        temp = [word for word in words if word not in vocab]
        vocab += temp

    return vocab

# 训练词向量
def train_word2vec_model(corpus, dim):
    # n>8.33logN <-- 词向量维度计算N词表长度 , min_count表示只有大于该阈值的词才会被纳入模型训练中(大数据情况下合理设置高阈值)
    model = Word2Vec(corpus, vector_size=dim, sg=1, window=5, min_count=4, workers=4)
    model.save(r"./model.w2v")
    return model

# 计算句子的词向量
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in sentence:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(sentence))
    return np.array(vectors)

# 计算带tf-idf权重的词向量
def sentence_to_tfidf_weighted_vector(model, sentence, idfs):
    vec_all = []
    total_tfidf_weight = 0

    for words in sentence:
        vec = np.zeros(model.vector_size)
        count = 0
        for word in words:
            if word in model.wv and word in idfs:
                count += 1
                tfidf_weight = idfs[word]
                vec += model.wv[word] * tfidf_weight
                total_tfidf_weight += tfidf_weight

        # 检查是否有有效的词
        if total_tfidf_weight > 0:
            vec /= total_tfidf_weight
        vec_all.append(vec)

    return np.array(vec_all)

def encode_sentence(text, vocab):
    input_id = []
    for char in text:
        input_id.append(vocab.get(char, vocab["[UNK]"]))
    input_id = padding(input_id)
    return input_id

# 补齐或截断输入的序列，使其可以在一个batch内运算
def padding(input_id):
    input_id = input_id[:30]
    input_id += [0] * (30 - len(input_id))
    return input_id

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

def tokenize(text):
    return jieba.cut(text)

def process_cut_words_to_vec(path, is_weight=1):
    # 1、读取数据
    df = load_csv_data(path)
    # 2、jieba分词并去除停用词
    stop_words = load_stop_words(r'../../week5/hit_stopwords.txt')
    x_cut_words = [[word for word in jieba.cut(data) if word not in stop_words] for data in df['review']]
    # 3、x_cut_words去重
    vocab = build_vocab(x_cut_words)
    # 4、训练、保存、加载word2vec词向量
    # 训练Word2Vec模型
    # w2v_model = train_word2vec_model(x_cut_words, 128)
    w2v_model = Word2Vec.load(r"./model.w2v")
    # 5、文本数据通过word2vec模型转换为词向量
    vectors_x = sentences_to_vectors(x_cut_words, w2v_model)
    # 6、带tfidf权重的词向量
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words=stop_words)  # 创建TfidfVectorizer实例
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['review'])
    # 获取词汇表和逆文档频率（IDF）
    vocab_vec = tfidf_vectorizer.get_feature_names_out()
    idfs = dict(zip(vocab_vec, tfidf_vectorizer.idf_))
    vectors_tfidf_x = sentence_to_tfidf_weighted_vector(w2v_model, x_cut_words, idfs)
    # 6、划分数据集
    x_train, x_valid, y_train, y_valid = train_test_split(vectors_x, df['label'].values, test_size=0.2, random_state=42)
    # x_train_weight, x_test_weight, y_train_weight, y_test_weight = (
    #     train_test_split(tfidf_matrix, df['label'], test_size=0.2, random_state=42))  # 效果不好
    x_train_weight, x_valid_weight, y_train_weight, y_valid_weight = (
        train_test_split(vectors_tfidf_x, df['label'].values, test_size=0.2, random_state=42))
    print(f'训练数据有{len(x_train)}条数据, 测试数据有{len(x_valid)}条数据')

    dataset_train_0 = CustomDataset(x_train, y_train)  # 默认不带tf-idf权重的训练数据
    dataset_test_0 = CustomDataset(x_valid, y_valid)
    dataloader_train_0 = DataLoader(dataset_train_0, batch_size=64)
    dataloader_test_0 = DataLoader(dataset_test_0, batch_size=64)

    if is_weight:
        dataset_train_1 = CustomDataset(x_train_weight, y_train_weight)
        dataset_test_1 = CustomDataset(x_valid_weight, y_valid_weight)
        dataloader_train_1 = DataLoader(dataset_train_1, batch_size=64)
        dataloader_test_1 = DataLoader(dataset_test_1, batch_size=64)
        return dataloader_train_1, dataloader_test_1

    return dataloader_train_0, dataloader_test_0

def process_segment_to_dataset(path):
    data = []
    df = load_csv_data(path)
    vocab = load_vocab(r'../chars.txt')
    for index, row in df.iterrows():
        # 按列名访问数据（推荐）
        review = row['review']
        input_id = encode_sentence(review, vocab)
        data.append(input_id)

    x_train, x_valid, y_train, y_valid = train_test_split(data, df['label'], test_size=0.2, random_state=42, shuffle=True)
    train_data_set = DataLoader(CustomDataset(np.array(x_train), np.array(y_train)), batch_size=32, shuffle=True)
    valid_data_set = DataLoader(CustomDataset(np.array(x_valid), np.array(y_valid)), batch_size=32, shuffle=True)

    return train_data_set, valid_data_set

dataloader_train, dataloader_test = process_cut_words_to_vec(r'../e-commerce_comments.csv', is_weight=1)