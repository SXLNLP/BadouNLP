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

sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def sentence_cut(sentence, Dict):
    results = []  # 存储切分结果
    path = []

    def find(start):
        if start == len(sentence):
            results.append(path[:])  # 复制当前路径
            return

        # 尝试所有可能的切分长度
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]  # 当前候选词
            if word in Dict:
                path.append(word)
                find(end)
                path.pop()

    find(0)  # 从位置0开始切分
    return results


# 测试
target = sentence_cut(sentence, Dict)
for result in target:
    print(result)
