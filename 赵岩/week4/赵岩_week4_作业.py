#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    def backtrack(start, path):
        # 如果已经处理到字符串的末尾，保存当前路径
        if start == len(sentence):
            result.append(path[:])
            return
        # 尝试从当前位置开始的所有可能的子串
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 如果子串在词典中，则继续递归处理剩余部分
            if word in Dict:
                path.append(word)  # 选择当前词
                backtrack(end, path)  # 递归处理剩余部分
                path.pop()  # 回溯，撤销选择

    result = []  # 用于存储所有可能的切分结果
    backtrack(0, [])  # 从字符串的起始位置开始递归
    return result

# 调用函数
all_possible_cuts = all_cut(sentence, Dict)

# 计算每个切分结果的总词频
def calculate_total_frequency(cut):
    return sum(Dict[word] for word in cut)

# 按照词频从高到低排序
all_possible_cuts.sort(key=calculate_total_frequency, reverse=True)

# 输出结果
for cut in all_possible_cuts:
    print(cut, "总词频:", calculate_total_frequency(cut))
