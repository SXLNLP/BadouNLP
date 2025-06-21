import jieba

class TextTokenizer:
    def __init__(self):
        """初始化TextTokenizer类"""
        pass
        
    def tokenize(self, text):
        """对文本进行分词
        
        Args:
            text (str): 待分词的文本
            
        Returns:
            list: 分词后的词列表
        """
        if not text:
            return []
            
        # 使用jieba进行分词
        words = jieba.lcut(text)
        
        # 过滤掉空字符串和单个字符
        words = [word for word in words if len(word.strip()) > 1]
        
        return words 