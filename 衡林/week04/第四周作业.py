# week3作业
# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
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

sentence = "经常有意见分歧"


def all_cut(sentence, word_dict):
    results = []  # 存储所有切分结果
    path = []  # 当前切分路径

    def backtrack(start_index):
        # 如果已切分到句子末尾，将当前路径加入结果
        if start_index == len(sentence):
            results.append(path[:])
            return

        # 尝试所有可能的结束位置
        for end_index in range(start_index + 1, len(sentence) + 1):
            word = sentence[start_index:end_index]
            # 如果当前子串在词典中
            if word in word_dict:
                path.append(word)  # 选择当前词
                backtrack(end_index)  # 递归处理剩余部分
                path.pop()  # 回溯，撤销选择

    backtrack(0)
    return results


# 测试
target = all_cut(sentence, Dict)
for cut in target:
    print(cut)
