

def fully_segment(sentence, word_dict):
    word_set = set(word_dict)
    max_word_length = max(len(word) for word in word_dict) if word_dict else 1
    n = len(sentence)
    result = []

    def backtrack(start, path):
        if start == n:
            result.append(path)
            return
        for length in range(1, max_word_length + 1):
            end = start + length
            if end > n:
                break
            candidate = sentence[start:end]
            if candidate in word_set:
                backtrack(end, path + [candidate])

    backtrack(0, [])
    return result


if __name__ == "__main__":
    word_dict = {"经常": 0.1, "经": 0.05, "有": 0.1, "常": 0.001, "有意见": 0.1,
                 "歧": 0.001, "意见": 0.2, "分歧": 0.2, "见": 0.1, "意": 0.05,
                 "见分歧": 0.05, "分": 0.1}
    sentence = "经常有意见分歧"

    segments = fully_segment(sentence, word_dict)

    for i, seg in enumerate(segments):
        print(f"切分方案{i+1}: {'/'.join(seg)}")
