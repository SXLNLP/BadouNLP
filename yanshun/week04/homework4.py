#week4作业

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
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
#递归实现
def all_cut(sentence, Dict):
    target = []
    if len(sentence) == 0:
        return [[]]

    for i in range(1, len(sentence)+1):
        if sentence[:i] in Dict.keys():
            for _ in all_cut(sentence[i:], Dict):
                target.append([sentence[:i]]+_)
    return target

#dp实现
def all_cut1(sentence, Dict):
    dp = [[] for i in range(len(sentence)+1)]
    dp[len(sentence)] = [[]]
    for i in range(len(sentence)-1,-1,-1):
        for j in range(len(sentence),i,-1):
            if sentence[i:j] in Dict.keys():
                for k in range(len(dp[j])):
                    dp[i].append([sentence[i:j]] + dp[j][k])
    target = dp[0]

    return target

#bfs实现
def all_cut2(sentence, Dict):
    end_queue = [0]
    cut_queue = [[]]
    target = []

    while end_queue:
        start = end_queue.pop(0)
        cut = cut_queue.pop(0)
        for end in range(start+1, len(sentence)+1):
            if sentence[start:end] in Dict.keys():
                if end == len(sentence):
                    target.append(cut + [sentence[start:end]])
                else:
                    end_queue.append(end)
                    cut_queue.append(cut + [sentence[start:end]])

    return target

#dfs实现
def all_cut3(sentence, Dict):
    stack = [0]
    path = [[]]
    target = []

    while stack:
        start = stack.pop(-1)
        cut = path.pop(-1)
        for end in range(start + 1, len(sentence) + 1):
            if sentence[start:end] in Dict.keys():
                if end == len(sentence):
                    target.append(cut + [sentence[start:end]])
                else:
                    stack.append(end)
                    path.append(cut + [sentence[start:end]])

    return target

def print_result(text):
    print("[")
    for _ in text:
        print(f"{_},")
    print("]")
    return

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

print_result(all_cut(sentence, Dict))
"""
[
['经', '常', '有', '意', '见', '分', '歧'],
['经', '常', '有', '意', '见', '分歧'],
['经', '常', '有', '意', '见分歧'],
['经', '常', '有', '意见', '分', '歧'],
['经', '常', '有', '意见', '分歧'],
['经', '常', '有意见', '分', '歧'],
['经', '常', '有意见', '分歧'],
['经常', '有', '意', '见', '分', '歧'],
['经常', '有', '意', '见', '分歧'],
['经常', '有', '意', '见分歧'],
['经常', '有', '意见', '分', '歧'],
['经常', '有', '意见', '分歧'],
['经常', '有意见', '分', '歧'],
['经常', '有意见', '分歧'],
]
"""
print_result(all_cut1(sentence, Dict))
"""
[
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
['经', '常', '有', '意', '见', '分', '歧'],
]
"""
print_result(all_cut2(sentence, Dict))
"""
[
['经常', '有意见', '分歧'],
['经', '常', '有意见', '分歧'],
['经常', '有', '意', '见分歧'],
['经常', '有', '意见', '分歧'],
['经常', '有意见', '分', '歧'],
['经', '常', '有', '意', '见分歧'],
['经', '常', '有', '意见', '分歧'],
['经', '常', '有意见', '分', '歧'],
['经常', '有', '意', '见', '分歧'],
['经常', '有', '意见', '分', '歧'],
['经', '常', '有', '意', '见', '分歧'],
['经', '常', '有', '意见', '分', '歧'],
['经常', '有', '意', '见', '分', '歧'],
['经', '常', '有', '意', '见', '分', '歧'],
]
"""
print_result(all_cut3(sentence, Dict))
"""
[
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
['经', '常', '有', '意', '见', '分', '歧'],
]
"""
