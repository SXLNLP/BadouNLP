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
def all_cut(s, Dict):
    if not s:
        return [[]]
    origin_target = []
    # 先找到字典中最大的字符尺寸
    max_length = max(len(word) for word in Dict)
    # 从第0个到第i个切分字符，但是最大切到字典里最大字符那里
    for i in range(1, max_length + 1):
        word = s[:i]
        # 在字典里就切下来，然后对后面的再用刚才的函数切
        if word in Dict:
            remaining_s = all_cut(s[i:], Dict)
            for cut in remaining_s:
                origin_target.append([word] + cut)
    # 假如第一个字符不在字典里，就直接切一个，然后对后面的字符继续使用这个函数
    if not origin_target and len(s) >= 1:
        origin_target = [[s[0]] + cut for cut in all_cut(s[1:], Dict)]

    target = []
    # 转元组去重
    seen = set()
    for seg in origin_target:
        seg_tuple = tuple(seg)
        if seg_tuple not in seen:
            seen.add(seg_tuple)
            target.append(seg)

    return target

def calculate_probability(target, Dict):
    prob_of_target = []
    for single_sort in target:
        total_prob = sum(Dict.get(word, 0.0) for word in single_sort)
        prob_of_target.append((single_sort, total_prob))
    # 降序排列
    prob_of_target.sort(key=lambda x: x[1], reverse=True)
    return prob_of_target
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

answer = all_cut(sentence, Dict)
probs = calculate_probability(answer, Dict)
for i, j in probs:
    print(f"{i}(概率：{j:.3f})")
