from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TextVectorizer:
    def __init__(self):
        """初始化TextVectorizer类"""
        self.vectorizer = TfidfVectorizer()
        
    def fit_transform(self, texts):
        """对文本列表进行TF-IDF向量化
        
        Args:
            texts (list): 文本列表，每个元素是分词后的词列表
            
        Returns:
            numpy.ndarray: TF-IDF矩阵
        """
        if not texts:
            return np.array([])
            
        # 将分词后的词列表转换为空格分隔的字符串
        text_strings = [' '.join(text) for text in texts]
        
        # 进行TF-IDF向量化
        tfidf_matrix = self.vectorizer.fit_transform(text_strings)
        
        return tfidf_matrix.toarray()
        
    def get_feature_names(self):
        """获取特征词列表
        
        Returns:
            list: 特征词列表
        """
        return self.vectorizer.get_feature_names_out() 