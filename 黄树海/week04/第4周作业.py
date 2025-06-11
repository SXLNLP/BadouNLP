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

sentence = "经常有意见分歧"

def all_cut(sentence, Dict):
    result = []
    def dfs(start, path):
        if start == len(sentence):
            result.append(path[:])
            return
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                dfs(end, path)
                path.pop()
    dfs(0, [])
    return result

# 获取所有可能的切分组合
all_segments = all_cut(sentence, Dict)
# 反转结果  
reversed_segments = all_segments[::-1]
print(reversed_segments)
