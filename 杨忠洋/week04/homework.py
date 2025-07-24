#week3作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
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
        "分": 0.1}

#待切分文本
sentence_for_cut = "经常有意见分歧"


#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence: str, dict: dict) -> list:
    max_word_len = max(len(word) for word in dict.keys())

    return backtrack(sentence, dict, {}, 0, max_word_len)


def backtrack(sentence: str, dict: dict, memo: dict, start: int, max_word_len: int) -> list:
    """
    递归实现全切分
    :param sentence: 待切分的文本
    :param dict: 词典
    :param memo: 避免重复计算 key: 起始位置, value: 起始位置后，已经计算过的子文本的切分结果
    :param start: 当前处理的起始位置
    :param max_word_len: 单词最大长度
    :return:
    """
    if start in memo:
        return memo[start]
    if start == len(sentence):
        return [[]]

    results = []
    for end in range(start + 1, min(start + max_word_len + 1, len(sentence) + 1)):
        word = sentence[start:end]
        if word in dict:
            for sub_path in backtrack(sentence, dict, memo, end, max_word_len):
                results.append([word] + sub_path)

    memo[start] = results
    return results


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
