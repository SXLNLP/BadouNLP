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

def cut_word(sentence, max_word_len):
    if len(sentence) == 0:
        return [[]]
    
    result = []
    for i in range(1, min(len(sentence) + 1, max_word_len + 1)):
        word = sentence[:i]
        for split in cut_word(sentence[i:], max_word_len):
            result.append([word] + split)
    # 返回所有切分结果      
    return result
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    max_word_len = max(len(word) for word in Dict.keys())
    results = cut_word(sentence, max_word_len)

    final_result = []

    # 筛选出所有在词典中的切分结果
    for result in results:
        flag = True
        for word in result:
            if word not in Dict:
                flag = False
                break
            
        if flag:
            final_result.append(result)
    
    return final_result

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

results = all_cut(sentence, Dict)
for result in results:
    print(result)
print(len(results))
