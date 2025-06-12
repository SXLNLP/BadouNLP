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


def all_cut(sentence, Dict):
    res = []  # 存储切分结果
    n = len(sentence)

    def dfs(start, path):
        # 设置递归出口
        if start == n:
            # 保存路径
            res.append(path.copy())  # 对于字符串这种不可变类型浅拷贝不影响
            return

        # 从当前位置开始遍历所有切分
        for end in range(start + 1, n + 1):
            word = sentence[start:end]
            # 当前分词存在于词典中，切分剩余部分
            if word in Dict:
                path.append(word)  # 选择当前词
                dfs(end, path)  # 递归处理剩余部分
                path.pop()  # 深度优先，到头了回退1层

    dfs(0, [])  # 从0开始
    return res


target = all_cut(sentence, Dict)

# 输出结果
for _, cut in enumerate(target):
    print(f"{cut}")
