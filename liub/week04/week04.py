# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/6/13
# @Author      : liuboyuan
# 字符串全切分

sentence = "经常有意见分歧"
word_dict = {"经常":0.1,
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

def word_break_with_dict(s, word_dict):
    result = []
    word_set = set(word_dict.keys())  # 只保留词典中的词
    stack = [(0, [])]  # 栈元素是 (当前位置, 当前路径)

    while stack:
        start, path = stack.pop()

        if start == len(s):
            result.append(path)
            continue

        for end in range(start + 1, len(s) + 1):
            word = s[start:end]
            if word in word_set:
                stack.append((end, path + [word]))  # 新路径加入栈

    return result

if __name__ == "__main__":
    res = word_break_with_dict(sentence, word_dict)
    print(res)