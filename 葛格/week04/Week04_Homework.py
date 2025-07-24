"""
给你一个字典，一句话，按照字典，把整句话所有可能的分词结果都写出来

回溯：每次选择1，2，。。。len(s)个字符，
剩下的直接递归，获取所有可能的结果

中间优化步骤可以用缓存法，把之前走过的记录下来

还写了一个暴力方法，用来验证
"""

from itertools import product


def all_cut(sentence, word_dict):
    word_set = set(word_dict.keys())
    mem = {}
    result = find_all(0, sentence, mem, word_set)
    for res in result:
        print(res)
    return result


# 从start位置开始,找到所有【start：end】的结果
def find_all(start, sentence, mem, word_set):
    if start == len(sentence):
        return [[]]
    if start in mem:
        return mem[start]

    result = []
    for end in range(start, len(sentence) + 1):
        word = sentence[start: end]
        if word in word_set:
            for back_word in find_all(end, sentence, mem, word_set):
                result.append([word] + back_word)
    mem[start] = result
    return result

# 暴力切除
def force_all_cut(sentence, word_dict):
    word_set = set(word_dict)
    n = len(sentence)
    results = []

    # 一共有 n - 1 个切分点，每个点可以切(1)或不切(0)
    # 枚举所有切法，比如 "我爱你" 有 2 个切分点 => 2^2 = 4 种切法
    for cut_mask in product([0, 1], repeat=n - 1):
        # 根据切法把句子切开
        split = []
        last_index = 0
        for i, cut in enumerate(cut_mask):
            if cut:
                split.append(sentence[last_index:i + 1])
                last_index = i + 1
        split.append(sentence[last_index:])

        # 检查每段是否都在字典里
        if all(word in word_set for word in split):
            results.append(split)

    return results


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
result1 = all_cut(sentence, Dict)
result2 = force_all_cut(sentence, Dict)
print(sorted(result1) == sorted(result2))
