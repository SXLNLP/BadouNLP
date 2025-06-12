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

#实现全切分函数，输出字符串根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    max_len = 1
    for key in Dict.keys():
        max_len = max(max_len, len(key))

    result = []
    def dfs(sentence, path):
        if not sentence: 
            result.append(path.copy())
            return
        
        for i in range(1, min(max_len, len(sentence)) + 1):
            word = sentence[:i]
            if word in Dict:
                path.append(word)
                dfs(sentence[i:], path)
                path.pop()

    dfs(sentence, [])
    return result

def calculate_score(segmentation, Dict):
    score = 0
    for word in segmentation:
        score += Dict[word]
    return score

result = all_cut(sentence, Dict)
print("\n全部切分结果如下：\n")

for ans in result:
    print(','.join(ans))

best_segmentation = max(result, key=lambda x: calculate_score(x, Dict))
print(f"\n其中最优切分方式为: {','.join(best_segmentation)}")

#目标输出;顺序不重要
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

