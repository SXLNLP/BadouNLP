# #week3作业

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

    # 使用DFS回溯生成切分方案
    def backtrack(start, path):
        # 如果已切分到句子末尾，将当前路径加入结果
        if start == len(sentence):
            results.append(path.copy())
            return

        # 尝试所有可能的结束位置
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 如果当前子串在词典中，则继续递归
            if word in word_dict:
                path.append(word)  # 选择当前词
                backtrack(end, path)  # 递归处理剩余部分
                path.pop()  # 撤销选择（回溯）

    backtrack(0, [])
    return results


# 获取全切分结果
cuts = all_cut(sentence, Dict)

# 打印所有切分方式
for i, cut in enumerate(cuts):
    print(f"切分方式{i + 1}: {cut}")
