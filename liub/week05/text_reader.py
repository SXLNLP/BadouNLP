class TextReader:
    def __init__(self, file_path):
        """初始化TextReader类
        
        Args:
            file_path (str): 文本文件路径
        """
        self.file_path = file_path
        
    def read(self):
        """读取文本文件
        
        Returns:
            str: 文件内容
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return None 