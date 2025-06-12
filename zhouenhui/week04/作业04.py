#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

#待切分文本

sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []  # 存储所有切分结果

    # 内部递归函数
    def dfs(start, path):
        # 当遍历到句子末尾时，将当前路径加入结果
        if start == len(sentence):
            target.append(path[:])  # 使用切片复制当前路径
            return

        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]  # 当前候选词
            # 如果词在词典中，继续递归切分剩余部分
            if word in Dict:
                path.append(word)  # 将当前词加入路径
                dfs(end, path)  # 递归处理剩余部分
                path.pop()  # 回溯，移除当前词

    dfs(0, [])
    return target


# 获取所有切分结果，并打印结果
target = all_cut(sentence, Dict)

for i, cut in enumerate(target):
    print(f"{i + 1}. {cut}")
