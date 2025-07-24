# loader.py
import re
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from config import global_config as config
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class TextDataset:
    def __init__(self):
        self.vector_model = None
        self.classifier = None
        self.vocab = None

    def clean_text(self, text):
        """清洗文本"""
        if not isinstance(text, str):
            return ""
        # 移除特殊字符，保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fa5\w\s，。！？；："\'、]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text):
        """中文分词"""
        return list(jieba.cut(text))

    def preprocess_data(self, df):
        """预处理数据"""
        # 清洗文本
        df['cleaned_text'] = df[config.text_column].apply(self.clean_text)

        # 分词
        df['tokens'] = df['cleaned_text'].apply(self.tokenize)

        # 构建词汇表
        all_tokens = [token for tokens in df['tokens'] for token in tokens]
        self.vocab = Counter(all_tokens)

        return df

    def train_fasttext_vectors(self, tokens_list):
        """训练FastText词向量"""
        # 训练FastText模型
        model = FastText(
            vector_size=config.vector_size,
            window=config.window,
            min_count=config.min_count,
            workers=config.workers,
            sg=config.sg,
            hs=config.hs,
            negative=config.negative,
            epochs=config.epochs
        )

        # 构建词汇表
        model.build_vocab(corpus_iterable=tokens_list)

        # 训练词向量
        model.train(
            corpus_iterable=tokens_list,
            total_examples=model.corpus_count,
            epochs=model.epochs
        )

        return model

    def text_to_vector(self, tokens, model, method='average'):
        """将文本转换为向量"""
        if method == 'average':
            # 平均词向量
            vectors = [model.wv[token] for token in tokens if token in model.wv]
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(config.vector_size)
        elif method == 'doc2vec':
            # 使用Doc2Vec风格
            return model.infer_vector(tokens)
        else:
            raise ValueError("未知的向量化方法")

    def prepare_classifier_data(self, df, vector_model):
        """准备分类器数据"""
        # 将文本转换为向量
        df['vector'] = df['tokens'].apply(
            lambda tokens: self.text_to_vector(tokens, vector_model))

        # 分割特征和标签
        X = np.array(df['vector'].tolist())
        y = df[config.label_column].values

        return X, y

    def train_classifier(self, X_train, y_train):
        """训练分类器"""
        # 这里可以选择不同的分类器
        classifier = LogisticRegression(
            max_iter=config.classifier_epochs,
            solver='saga',
            penalty='l2',
            C=1.0 / config.classifier_lr
        )

        # 训练分类器
        classifier.fit(X_train, y_train)

        return classifier

    def load_and_process_data(self):
        """加载并处理数据"""
        # 加载数据
        try:
            df = pd.read_csv(config.data_path)
        except Exception as e:
            raise FileNotFoundError(f"加载数据失败: {e}")

        # 检查列名
        if config.text_column not in df.columns or config.label_column not in df.columns:
            # 尝试自动识别列
            text_col = df.columns[0]
            label_col = df.columns[1]
            df = df.rename(columns={
                text_col: config.text_column,
                label_col: config.label_column
            })

        # 确保标签是整数
        df[config.label_column] = df[config.label_column].astype(int)

        # 预处理数据
        df = self.preprocess_data(df)

        # 划分训练集和验证集
        train_df, val_df = train_test_split(
            df, test_size=config.test_size, random_state=42,
            stratify=df[config.label_column])

        # 训练FastText词向量模型
        print("训练FastText词向量...")
        self.vector_model = self.train_fasttext_vectors(train_df['tokens'])

        # 保存词向量模型
        self.vector_model.save(config.vector_model_path)
        print(f"词向量模型已保存至 {config.vector_model_path}")

        # 准备分类器数据
        print("准备分类器数据...")
        X_train, y_train = self.prepare_classifier_data(train_df, self.vector_model)
        X_val, y_val = self.prepare_classifier_data(val_df, self.vector_model)

        # 训练分类器
        print("训练分类器...")
        self.classifier = self.train_classifier(X_train, y_train)

        # 保存分类器
        import joblib
        joblib.dump(self.classifier, config.classifier_model_path)
        print(f"分类器模型已保存至 {config.classifier_model_path}")

        return train_df, val_df, X_train, y_train, X_val, y_val

    def analyze_data(self):
        """数据分析"""
        try:
            df = pd.read_csv(config.data_path)
        except Exception as e:
            print(f"加载数据失败: {e}")
            return {}

        # 检查列名
        if config.text_column not in df.columns or config.label_column not in df.columns:
            text_col = df.columns[0]
            label_col = df.columns[1]
            df = df.rename(columns={
                text_col: config.text_column,
                label_col: config.label_column
            })

        # 正负样本分析
        positive_count = (df[config.label_column] == 1).sum()
        negative_count = (df[config.label_column] == 0).sum()

        # 文本长度分析
        df['cleaned_text'] = df[config.text_column].apply(self.clean_text)
        text_lengths = df['cleaned_text'].apply(len)  # 字符长度

        return {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'avg_length': text_lengths.mean(),
            'max_length': text_lengths.max(),
            'min_length': text_lengths.min()
        }