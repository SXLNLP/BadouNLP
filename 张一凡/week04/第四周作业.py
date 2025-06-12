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
sentence = "经常有意见分歧"


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    # 计算字典中最长词的长度，用于限制切分长度
    max_len = max(len(word) for word in Dict) if Dict else 0
    results = []

    # 递归回溯函数
    def backtrack(start, path):
        # 如果已切分到句子末尾，将当前路径加入结果
        if start == len(sentence):
            results.append(path[:])
            return

        # 尝试所有可能的切分位置（从当前开始到最长词长度）
        end = min(start + max_len, len(sentence))
        for i in range(start + 1, end + 1):
            word = sentence[start:i]
            # 如果当前子串在字典中，则继续递归切分剩余部分
            if word in Dict:
                path.append(word)
                backtrack(i, path)
                path.pop()  # 回溯，移除最后添加的词

    backtrack(0, [])
    return results


# 测试
target = all_cut(sentence, Dict)
for t in target:
    print(t)
