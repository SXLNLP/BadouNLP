my_dict = {
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


def all_cut(sentence, my_dict):
    result = []
    n = len(sentence)

    def backtrack(start, path):
        if start == n:
            result.append(path.copy())
            return
        for end in range(start + 1, n + 1):
            word = sentence[start:end]
            if word in my_dict:
                path.append(word)
                backtrack(end, path)
                path.pop()

    backtrack(0, [])

    # 去重处理
    unique_result = []
    seen = set()
    for item in result:
        tuple_item = tuple(item)
        if tuple_item not in seen:
            seen.add(tuple_item)
            unique_result.append(list(item))
    print(unique_result)



if __name__ == '__main__':
    all_cut(sentence, my_dict)

target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '岐'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '岐'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '岐'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '岐'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '岐'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '岐']
]
