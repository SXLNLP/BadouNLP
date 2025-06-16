def all_cut(sentence, Dict):
    if not sentence:
        return []
    # 构建有向无环图(DAG)
    n = len(sentence)
    dag = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n + 1):
            word = sentence[i:j]
            if word in Dict:
                dag[i].append(j)
    # 使用深度优先搜索(DFS)来获取所有可能的切分路径
    result = []
    path = []
    def dfs(start):
        if start == n:
            result.append(path[:])
            return
        for end in dag[start]:
            word = sentence[start:end]
            path.append(word)
            dfs(end)
            path.pop()
    dfs(0)
    return result
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
sentence = "经常有意见分歧"
result = all_cut(sentence, Dict)
# 输出结果
for seg in result:
    print(seg)
