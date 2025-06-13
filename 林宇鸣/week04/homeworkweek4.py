# week4 作业

# 词典：每个词后方存储的是其词频，词频仅为示例，不会用到，也可以自行修改
Dict = {"经常":0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"

# 实现全切分函数，输出根据字典能够切分出所有的切分方式
def all_cut(sentence, Dict):
    def dfs(start, path):
        # 如果到达字符串末尾，保存当前路径
        if start == len(sentence):
            result.add(tuple(path))  # 使用元组将路径转换为不可变类型以便加入集合
            return

        # 从当前起始位置遍历所有可能的结束位置
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]  # 获取当前切分的词
            if word in Dict:  # 判断词是否在字典中
                dfs(end, path + [word])  # 递归调用，继续切分后面的部分

    result = set()  # 使用集合来存储结果以避免重复
    dfs(0, [])
    return [list(words) for words in result]  # 将集合转换为列表

# 调用函数并输出结果
target = all_cut(sentence, Dict)

# 输出所有的切分情况
print(target)


# 目标输出: 顺序不重要
# target = [['经常', '有意见', '分歧']]

import jieba

# 测试分词功能
text = "我爱自然语言处理"
words = jieba.lcut(text)
print("/ ".join(words))
