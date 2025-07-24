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
    result = []

    def backtrack(start, path):
        # 如果已经到达句子末尾，记录当前分词结果
        if start == len(sentence):
            result.append(path[:])
            return

        # 尝试从当前位置开始的所有可能词语
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                # 选择当前词语
                path.append(word)
                # 递归处理剩余部分
                backtrack(end, path)
                # 回溯，移除当前选择
                path.pop()

    # 开始回溯搜索
    backtrack(0, [])
    return result

for i in all_cut(sentence, Dict):
    print(i, end="\n")
