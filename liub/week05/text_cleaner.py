import re

class TextCleaner:
    def __init__(self):
        """初始化TextCleaner类"""
        # 定义要删除的特殊字符
        self.special_chars = r'[【】\n。，、：；""''《》？！\\s]+'
        
    def clean(self, text):
        """清洗文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""
            
        # 删除特殊字符
        text = re.sub(self.special_chars, ' ', text)
        
        # 删除数字
        text = re.sub(r'\d+', '', text)
        
        # 删除英文字符
        text = re.sub(r'[a-zA-Z]+', '', text)
        
        # 删除多余的空格
        text = ' '.join(text.split())
        
        return text 