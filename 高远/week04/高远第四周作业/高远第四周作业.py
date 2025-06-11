# 高远第四周作业
# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

# 待切分文本
sentence = "常经有意见分歧"


# 实现全切分函数
def all_cut(sentence, Dict):
    # 递归函数：尝试从每个位置开始切分
    def recursive_cut(s):
        if not s:
            return [[]]  # 返回一个空列表，表示切分结束
        result = []
        # 尝试每一个可能的切割点
        for i in range(1, len(s) + 1):
            word = s[:i]
            if word in Dict:
                # 如果词存在字典中，递归处理剩下的部分
                rest_results = recursive_cut(s[i:])
                for rest in rest_results:
                    result.append([word] + rest)
        return result

    # 调用递归函数从头开始切分整个句子
    return recursive_cut(sentence)


# 获取所有的切分方式
target = all_cut(sentence, Dict)

# 输出结果
for seq in target:
    print(seq)

# 输出
# ['常', '经', '有', '意', '见', '分', '歧']
# ['常', '经', '有', '意', '见', '分歧']
# ['常', '经', '有', '意', '见分歧']
# ['常', '经', '有', '意见', '分', '歧']
# ['常', '经', '有', '意见', '分歧']
# ['常', '经', '有意见', '分', '歧']
# ['常', '经', '有意见', '分歧']
