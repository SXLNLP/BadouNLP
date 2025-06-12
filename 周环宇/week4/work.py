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
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    def backtrack(start, path, result):
        #如果 start 等于句子长度，说明已经切分到句子末尾，将当前路径 path 加入结果 result
        if start == len(sentence):
            result.append(path.copy())
            return
        #否则,从start开始尝试所有可能的结束位置end,将子串sentence[start:end]作为候选词,如果候选词在字典中,将其加入path,并递归处理剩余部分
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                backtrack(end, path, result)
                path.pop()

    result = []
    backtrack(0, [], result)
    return result


# 运行全切分函数
target = all_cut(sentence, Dict)

# 打印所有切分方式
for cut in target:
    print(cut)

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

