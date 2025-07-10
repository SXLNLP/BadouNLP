# 词典
Dict = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.001,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1
}

sentence = "经常有意见分歧"
def all_cut(sentence, Dict):
    """
    对输入文本进行全切分，找出所有可能的切分方式

    参数:
    sentence -- 待切分的中文文本
    Dict -- 词典，键为词语，值为词频（本函数未使用）

    返回:
    result -- 包含所有切分方式的列表，每个切分方式是一个词语列表
    """
    result = []  # 存储所有切分结果

    # 定义递归函数进行切分
    def cut_recursive(start, current_split):
        # 如果已经切分到文本末尾，将当前切分方式加入结果
        if start == len(sentence):
            result.append(current_split.copy())
            return

        # 尝试从当前位置开始的所有可能词语
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 如果词语在词典中，则继续递归切分剩余部分
            if word in Dict:
                current_split.append(word)
                cut_recursive(end, current_split)
                current_split.pop()  # 回溯，移除最后添加的词语

    # 从文本起始位置开始切分
    cut_recursive(0, [])
    return result



all_partitions = all_cut(sentence, Dict)

# 输出所有切分方式
print(f"文本 '{sentence}' 的所有切分方式：")
for i, partition in enumerate(all_partitions, 1):
    print(f"{i}. {partition}")
