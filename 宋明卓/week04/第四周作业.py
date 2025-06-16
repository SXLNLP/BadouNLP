# week4作业
# 待切分文本
sentence = "经常有意见分歧"
# #实现全切分函数，输出根据字典能够切分出的所有的切分方式
word_dict = {"经常": 0.3,
             "经": 0.05,
             "有": 0.3,
             "常": 0.001,
             "有意见": 0.001,
             "歧": 0.001,
             "意见": 0.3,
             "分歧": 0.3,
             "见": 0.05,
             "意": 0.05,
             "见分歧": 0.05,
             "分": 0.1}


def all_cut(sentence, word_dict):
    # 使用回溯函数来返回所有可能的切分结果
    def backtrack(start):
        # 递归终止条件
        if start == len(sentence):
            return [[]]
        result = []
        # 从当时前位置开始尝试不同的切分效果
        for end in range(start, len(sentence)):
            word = sentence[start:end + 1]
            if word in word_dict:
                # 使用递归来切分剩余的句子
                next_cuts = backtrack(end + 1)
                for cut in next_cuts:
                    result.append([word] + cut)
        return result

    return backtrack(0)


if __name__ == '__main__':
    all_cuts = all_cut(sentence, word_dict)
    data = []
    for cut in all_cuts:
        print(cut)
        all_sum = 0
        for word in cut:
            sum = word_dict[word]
            all_sum += sum
        data.append(cut + [all_sum])
        data.sort(key=lambda x: x[-1], reverse=True)
    print(f'max_row-->{data[0]}')
