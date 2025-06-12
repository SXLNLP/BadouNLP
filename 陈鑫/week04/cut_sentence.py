# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

# 待切分文本
Sentence = "经常有意见分歧"

"""
# 目标输出;顺序不重要
target = [
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
"""


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, word_dict):
    def backtrack(start, current_path):
        # 如果已经处理到文本的末尾，将当前路径加入结果中
        if start == len(sentence):
            result.append(current_path[:])  # 注意浅拷贝（current_path[:]）和引用(current_path)的区别
            return

        # 尝试从当前索引开始的所有可能切分
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in word_dict:
                current_path.append(word)
                backtrack(end, current_path)
                current_path.pop()  # 回溯，移除最后一个词

    result = []
    backtrack(0, [])
    return result


res = all_cut(Sentence, Dict)
for item in res:
    print(item)
