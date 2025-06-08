vocab = {
    "经常": 0.1,
    "经": 0.05,
    "常": 0.001,
    "有": 0.1,
    "意见": 0.2,
    "意": 0.05,
    "见": 0.05,
    "分歧": 0.2,
    "分": 0.1,
    "见分歧": 0.05,
    "有意见": 0.1,
}

sentence = "常经有意见分歧"

def get_DAG(sentence, vocab):
    dag = {}
    N = len(sentence)
    for start in range(N):
        ends = []
        for end in range(start, N):
            word = sentence[start:end+1]
            if word in vocab:
                ends.append(end)
        if not ends:
            ends.append(start)  # 单字成词 fallback
        dag[start] = ends
    return dag

def dfs(sentence, dag, start, path, results):
    if start == len(sentence):
        results.append(path[:])
        return
    for end in dag[start]:
        word = sentence[start:end+1]
        path.append(word)
        dfs(sentence, dag, end + 1, path, results)
        path.pop()

def main():
    sentence = "常经有意见分歧"
    dag = get_DAG(sentence, vocab)
    results = []
    dfs(sentence, dag, 0, [], results)

    for path in results:
        print(path)

if __name__ == "__main__":
    main()
