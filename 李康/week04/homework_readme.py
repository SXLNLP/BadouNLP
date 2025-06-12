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

# 待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def dfs(cut, sub_cut, i, sentence, Dict):
    if i == len(sentence):
        sub_cut_copy = sub_cut.copy()
        cut.append(sub_cut_copy[:])
        return
    for j in range(i, len(sentence)):
        sub_str = sentence[i:j+1]
        if sub_str in Dict:
            sub_cut.append(sub_str)
            dfs(cut, sub_cut, j + 1, sentence, Dict)
            sub_cut.pop()

def all_cut(sentence, Dict):
    cut = []
    sub_cut = []
    dfs(cut, sub_cut, 0, sentence, Dict)

    return cut

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

if __name__ == '__main__':

    ans = all_cut(sentence, Dict)
    print(ans)

