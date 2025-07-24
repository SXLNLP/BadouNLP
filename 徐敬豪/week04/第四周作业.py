def generate_all_splits(sentence, word_dict, max_len):
    def backtrack(s, path, results):
        if not s:
            results.append(path.copy())
            return

        # 尝试所有可能的切分长度（从最长到最短）
        for l in range(min(max_len, len(s)), 0, -1):
            word = s[:l]
            if word in word_dict:
                path.append(word)
                backtrack(s[l:], path, results)
                path.pop()

        # 处理单字（如果单字在词典中）
        if len(s) >= 1 and s[0] in word_dict:
            path.append(s[0])
            backtrack(s[1:], path, results)
            path.pop()

    results = []
    backtrack(sentence, [], results)
    return results


# 词典数据
Dict = {
    "经常", "经", "有", "常",
    "有意见", "歧", "意见", "分歧",
    "见", "意", "见分歧", "分"
}

# 计算词典中的最大词长
max_len = max(len(word) for word in Dict)

# 要切分的句子
sentence = "经常有意见分歧"

# 生成所有可能的切分
all_splits = generate_all_splits(sentence, Dict, max_len)

# 去掉重复
unique_splits = []
seen = set()
for split in all_splits:
    tuple_split = tuple(split)
    if tuple_split not in seen:
        seen.add(tuple_split)
        unique_splits.append(list(tuple_split))

# 按切分数量和字典序排序
unique_splits.sort(key=lambda x: (len(x), str(x)))

# 打印所有切分
for i, split in enumerate(unique_splits):
    print(f"{i + 1}. {split}")
