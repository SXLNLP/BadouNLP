# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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


def all_cut(sentence, Dict):
    def backtrack(start, path):
        if start == len(sentence):
            results.append(path[:])
            return

        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start: end]
            if (word in Dict):
                path.append(word)
                backtrack(end, path)
                path.pop()

        return

    results = []
    backtrack(0, [])
    return results


def main():
    sentence = "经常有意见分歧"
    target = all_cut(sentence, Dict)
    print(target)

if __name__ == "__main__":
    main()
