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

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    return target

def all_cut(sentence, Dict):
    """
    全切分函数，返回所有可能的切分路径，词必须出现在 Dict 中。
    :param sentence: 待切分的字符串
    :param Dict: 词典，key 为词，value 为词频（不使用）
    :return: 所有切分路径（列表嵌套）
    """
    result = []  # 保存所有切分路径

    def backtrack(start, path):
        # 递归出口：到达句尾
        if start == len(sentence):
            result.append(path[:])  # 拷贝当前路径加入结果
            return
        
        # 遍历从当前位置开始的所有子串
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)           # 选择当前词
                backtrack(end, path)        # 递归处理剩余部分
                path.pop()                  # 回溯，撤销上一步选择

    backtrack(0, [])
    return result


#目标输出;顺序不重要
target = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]

target = all_cut(sentence, Dict)
print(target)
