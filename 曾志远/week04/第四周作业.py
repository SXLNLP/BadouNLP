#week4作业

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

# 待切分文本
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence: str, Dict: dict) -> list:
    # TODO
    target = []
    char_len_list = []
    # 词表中词的长度
    for char in Dict:
        char_len_list.append(len(char))
    # 去重
    char_len_set = set(char_len_list)
    cutList(0, [], target, sentence, char_len_set)
    return target


def cutList(start: int, path: list, target: list, sentence: str, char_len_set: set) -> None:
    if start == len(sentence):  # 终止条件
        target.append(path[:])
        return
    for i in range(start, len(sentence)):
        substr = sentence[start:i + 1]  # 当前子串 s[start..i]
        if substr in Dict.keys() and len(substr) in char_len_set:  # 筛选符合条件的子串 (剪枝)
            path.append(substr)  # 做出选择
            cutList(i + 1, path, target, sentence, char_len_set)  # 递归切割剩余部分
            path.pop()  # 撤销选择（回溯）


target = all_cut(sentence, Dict)
for i in range(len(target)):
    print(target[i])


#目标输出;顺序不重要
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
