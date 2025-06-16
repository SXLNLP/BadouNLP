#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence_for_cut = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence:str, dict:dict) -> list:
    # 获取最大词长
    max_word_len = 0
    for word in dict.keys():
        max_word_len = max(max_word_len, len(word))
    target = []

    return backtrack(sentence, dict, max_word_len, 0, [], target)


def backtrack(sentence: str, dict: dict, max_word_len: int, start: int, path: list, target:list) -> list:
    """
    使用回溯算法生成所有可能的中文分词组合

    :param sentence:需要分词的中文句子
    :param dict: 词典集合，包含所有合法词汇
    :param max_word_len: 词典中最长词汇的长度
    :param start: 当前处理的起始位置
    :param path: 当前已选择的分词路径
    :param target: 存储所有有效分词结果的列表

    :return list: 包含所有有效分词组合的列表，每个元素是一个分词路径
    """
    if start == len(sentence):
        # 递归终止条件：当处理完所有字符时，将当前分词路径添加到结果中
        target.append(path.copy())
        return target
    for length in range(1, max_word_len + 1):
        # 尝试不同长度的单词分割
        end = start + length
        if end > len(sentence):
            break
        word = sentence[start:end]
        if word in dict.keys():
            # 选择当前单词并递归处理后续字符
            path.append(word)
            backtrack(sentence, dict, max_word_len, end, path, target)
            path.pop()  # 回溯：撤销当前选择，尝试其他分割方式
    return target

print(all_cut(sentence_for_cut, Dict))

#目标输出;顺序不重要
target_std = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

# 验证结果与标准输出是否一致
result = all_cut(sentence_for_cut, Dict)
set_result = set(tuple(sublist) for sublist in result)
set_std = set(tuple(sublist) for sublist in target_std)
assert set_result == set_std, f"验证失败：\n实际结果：{set_result} \n预期结果：{set_std}"
print("验证通过！")

