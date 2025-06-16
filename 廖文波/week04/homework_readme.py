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
    result = []
    # 双重for循环找到存在于词典中的所有字串，并进行位置分组
    child_list = []
    for i in range(len(sentence)):
        child = []
        for j in range(i + 1, len(sentence) + 1):
            # 获取当前切分的子串
            substring = sentence[i:j]
            if(substring in Dict):
                child.append(substring)
        child_list.append(child)
    print("子串列表：", child_list)
     # 子串列表： [['经', '经常'], ['常'], ['有', '有意见'], ['意', '意见'], ['见', '见分歧'], ['分', '分歧'], ['歧']]
    # 从索引0开始进行切分
    dp(0, [], child_list, result)
    # 返回最终的切分结果
    return result
# 递归函数进行深度优先搜索切分
def dp(index, current_cut, child_list, result):
    if index == len(sentence):
        print("=================================")
        # 如果到达字符串末尾，将当前切分结果添加到结果列表中
        result.append(current_cut)
        return
    for word in child_list[index]:
        print(current_cut, "递归到：", word, "索引：", index)
        dp(index + len(word), current_cut + [word], child_list, result)


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

if __name__ == "__main__":
    result = all_cut(sentence, Dict)
    print(len(result))
    for r in result:
        print(r)

