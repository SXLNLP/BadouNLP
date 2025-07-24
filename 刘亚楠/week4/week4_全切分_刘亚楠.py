#week4作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, word_dict):
    # 预处理词典，构建带词频的逆向最大匹配表
    max_len = max(len(word) for word in word_dict) if word_dict else 1
    word_set = set(word_dict.keys())

    # 记忆化搜索缓存
    memo = {}

    def backtrack(start):

        if start in memo:
            return memo[start]
        if start == len(sentence):
            return [[]]

        results = []
        # 从最大可能长度开始尝试
        for length in range(min(max_len, len(sentence) - start), 0, -1):
            word = sentence[start:start + length]

            if word in word_set:
                for suffix in backtrack(start + length):
                    results.append([word] + suffix)

        memo[start] = results

        return results

    return backtrack(0)


# 测试调用
result = all_cut("经常有意见分歧", Dict)
for seq in result:
    print(seq)

#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

