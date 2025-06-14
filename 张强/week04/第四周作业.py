#week3作业

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


def load_word_dict(Dict):
    max_word_length = 0
    for key in Dict.keys():
        max_word_length = max(max_word_length, len(key))
        # print(max_word_length)
    return max_word_length

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict, max_len):
    results = []  # 存储所有切分结果
    current = []  # 当前切分路径

    def backtrack(start):
        """回溯切分函数"""
        if start == len(sentence):
            # 到达字符串末尾，保存当前切分
            results.append(current[:])
            # print(results)
            return

        # 尝试所有可能的切分长度（从最长到最短）
        lens = min(start + max_len, len(sentence))
        for i in range(lens, start, -1):
            word = sentence[start:i]
          

            # 如果单词在词典中，或长度=1（单字切分）
            if word in Dict or len(word) == 1:
                current.append(word)  # 选择当前切分
                # print(current)
                backtrack(i)  # 递归处理剩余部分
                current.pop()  # 回溯，撤销选择
              

    backtrack(0)  # 从位置0开始回溯
    return results

max_len = load_word_dict(Dict)
results = all_cut(sentence, Dict, max_len)
for i, seg in enumerate(results):
    print(f"切分方案 {i+1}: {seg}")
