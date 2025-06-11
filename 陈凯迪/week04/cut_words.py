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

sentence = "经常有意见分歧"  # 修正输入字符串为"经常有意见分歧"


def all_cut(sentence, Dict):
    results = []  # 存储所有切分结果
    path = []  # 当前切分路径

    def dfs(start):
        # 当切分位置到达句子末尾，将当前路径添加到结果中
        if start == len(sentence):
            results.append(path[:])  # 使用切片复制当前路径
            return

        # 尝试从start位置开始的所有可能子串
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 如果子串在词典中，则继续切分剩余部分
            if word in Dict:
                path.append(word)  # 将当前词加入路径
                dfs(end)  # 递归切分剩余部分
                path.pop()  # 回溯，移除当前词

    dfs(0)  # 从位置0开始深度优先搜索
    return results


# 获取所有切分结果
target = all_cut(sentence, Dict)

# 输出结果（与目标格式一致）
for cut in target:
    print(cut)
