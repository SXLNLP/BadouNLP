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
def all_cut(chars, dic):
    results = []
    # 获取词典最长词的长度
    max_words_length = max(len(word) for word in dic)
    # 使用递归方式 获取所有符合条件的数据结果
    generate_combinations(chars, dic, 0, [], results, max_words_length)
    return results


# 使用贪心算法
def generate_combinations(chars, dic, start, current_path, results, max_word_length):
    # 完成字符串切分
    if start == len(chars):
        results.append(list(current_path))
        return
    # 尝试所有的可能 ,从1 到最大词长
    for length in range(1, max_word_length + 1):
        if start + length > len(chars):
            break
        word = chars[start:start + length]
        # 判断是否子串在词典中
        if word in dic:
            # 将当前词加入到结果中
            current_path.append(word)
            # 继续递归处理剩余部分
            generate_combinations(chars, dic, start + length, current_path, results, max_word_length)
            #删除当前词,使用其他词
            current_path.pop()


sentence = "经常有意见分歧"

if __name__ == '__main__':
    words = all_cut(sentence, Dict)
    print(words)
