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
    def cut_string(s):
        for i in range(len(s)):
            if s[:i+1] in word_dict:
                if len(result) == 0 or result[-1][-1].find(s[-1]) != -1:
                    result.append([s[:i+1]])
                else:
                    result[-1].append(s[:i+1])
                cut_string(s[i+1:])

    result = []
    cut_string(Sentence)

    str_len = len(sentence)
    for i in range(len(result)):
        item_len = sum([len(x) for x in result[i]])
        if item_len != str_len:
            for idx in range(len(result[i-1])):
                result[i].insert(idx, result[i-1][idx])
                item_len += len(result[i-1][idx])
                if item_len >= str_len:
                    break
    return result


res = all_cut(Sentence, Dict)
for item in res:
    print(item)
