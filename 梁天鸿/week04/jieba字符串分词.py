import jieba

def full_segmentation_jieba(text):
    """使用jieba的全模式进行全切分"""
    return list(jieba.cut(text, cut_all=True))

# 示例
text = "我爱自然语言处理"
result = full_segmentation_jieba(text)
print("全切分结果:", result)
