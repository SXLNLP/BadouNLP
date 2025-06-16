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
def all_segmentations(sentence, Dict):
    n = len(sentence)
    dp = [[] for _ in range(n + 1)]#创建一个长度为n+1的列表，每个元素是一个空列表
    dp[0] = [[]]#初始化dp[0]为一个包含一个空列表的列表

    for i in range(1, n + 1):#外层循环遍历句子的每个位置（从1到n）
        for j in range(i): #内层循环检查所在可能的子串起始位置j（从0到i-1）
            word = sentence[j:i] #提取子串，从位置j到i-1的字符
            if word in Dict: #检查这个子串是否在字典中
                for seg in dp[j]:#如果在词典中，遍历前j个字符的所有分词方式
                    dp[i].append(seg + [word]) #将当前词添加到每个分词方式后面，形成新的分词组合
    print(dp[n])
    return dp[n]

def find_max_freq_segmentation(sentence, word_dict):
    all_segs = all_segmentations(sentence, word_dict)#生成所有可能的分词组合
    max_freq = -1  #初始化
    best_seg = None

    for seg in all_segs:  #遍历每个分词方案
        current_freq = sum(word_dict[word] for word in seg) #计算当前词频总和
        if current_freq > max_freq:
            max_freq = current_freq
            best_seg = seg

    return best_seg, max_freq


# 示例使用
sentence = "经常有意见分歧"
dict = {
    "经常": 0.1, "经": 0.05, "有": 0.1, "常": 0.001,
    "有意见": 0.1, "歧": 0.001, "意见": 0.2, "分歧": 0.2,
    "见": 0.05, "意": 0.05, "见分歧": 0.05, "分": 0.1
}

best_seg, max_freq = find_max_freq_segmentation(sentence, dict)
print("最佳切分:", best_seg)
print("最高总词频:", max_freq)

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
