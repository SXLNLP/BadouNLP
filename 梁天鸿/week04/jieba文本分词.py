import jieba

def segment_file_jieba(input_file, output_file, mode='default'):
    """
    使用jieba对文本文件进行分词
    
    参数:
    input_file: 输入文本文件路径
    output_file: 输出分词结果文件路径
    mode: 分词模式，'default'(精确模式),'full'(全模式),'search'(搜索引擎模式)
    """
    try:
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 根据指定模式进行分词
        if mode == 'full':
            words = jieba.cut(content, cut_all=True)
        elif mode == 'search':
            words = jieba.cut_for_search(content)
        else:  # 默认精确模式
            words = jieba.cut(content, cut_all=False)
        
        # 将分词结果写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("/ ".join(words))
        
        print(f"分词完成，结果已保存至 {output_file}")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")

# 使用示例
segment_file_jieba('input.txt', 'output_jieba.txt', mode='full')
