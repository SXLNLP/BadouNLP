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
    #TODO
    def preprocess(sentence):
        max_len = 3
        table = []
        for i in reversed(range(len(sentence))):
            list = []
            for j in range(min(i + max_len, len(sentence)), i, -1):
                if sentence[i:j] in Dict:
                    list.append(j - i)
            table.append(list)
        return table

    search_table = preprocess(sentence)
    segmentations = []

    def dfs(segment, string):
        if not string:
            segmentations.append(segment)
            return
        list = search_table[len(string) - 1]
        if list:
            for i in list:
                dfs(segment + [string[0:i]], string[i:])
        else:
            dfs(segment + [string[0:1]], string[1:])

    dfs([], sentence)
    return segmentations

print(all_cut(sentence,Dict))

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

