# week3 作业
# 词典，每个词后为词频（仅供选择最佳路径时使用）
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

# 待切分句子
sentence = "经常有意见分歧"


#实现全切分函数，输出根据字典能够切分出的所有的切分方式

def all_cut_dfs(sentence, Dict):
    target = []

    def dfs(index, path):
        if index == len(sentence):  
            target.append(path)
            return
        for end in range(index + 1, len(sentence) + 1):
            word = sentence[index:end]
            if word in Dict:  
                dfs(end, path + [word])

    dfs(0, [])
    return target


# 找出词频乘积最大的路径
def get_best_cut(cuts, Dict):
    best_path = None
    best_score = 0

    for cut in cuts:
        score = 1
        for word in cut:
            score *= Dict.get(word, 1e-8)  
        if score > best_score:
            best_score = score
            best_path = cut
    return best_path, best_score


# 运行主程序
print("搜索所有切法")
cuts1 = all_cut_dfs(sentence, Dict)

print("\n【所有切分方式】")
for c in cuts1:
    print(c)

print("\n【最佳切分路径】")
best_cut, best_score = get_best_cut(cuts1, Dict)
print("切分：", best_cut)
print("词频乘积得分：", best_score)


# 目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]
