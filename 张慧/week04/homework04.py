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


def all_cut(sentence, Dict):
    if not sentence:
        return [[]]

    results = []
    for i in range(1, len(sentence) + 1):
        word = sentence[:i]
        if word in Dict:
            for rest in all_cut(sentence[i:], Dict):
                results.append([word] + rest)

    return results


# 测试并打印结果
results = all_cut(sentence, Dict)
for i, cut in enumerate(results, 1):
    print(f"{i:2d}: {cut}")
    
