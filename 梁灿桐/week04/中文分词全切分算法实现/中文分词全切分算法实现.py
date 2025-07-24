# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
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
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"  # 修正为正确的句子顺序

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    results = []
    path = []

    def backtrack(start):
        # 当切分位置到达句子末尾，保存当前路径
        if start == len(sentence):
            results.append(path[:])
            return
        # 尝试所有可能的结束位置
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 如果当前词在词典中，继续递归切分剩余部分
            if word in Dict:
                path.append(word)
                backtrack(end)
                path.pop()
    backtrack(0)
    return results

# 调用函数并获取结果
target = all_cut(sentence, Dict)

# 输出所有切分结果
print(f"共有{len(target)}种切分方式：")
for i, seg in enumerate(target, 1):
    print(f"{i}: {seg}")
