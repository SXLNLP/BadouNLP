#week4作业

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
sentence = "常经有意见分歧"



#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []
    if sentence == "":
        return [[]]
    for i in range(1, len(sentence)+1):
        word = sentence[:i]
        if word in Dict:
            targets = all_cut(sentence[i:], Dict)
            for j in targets:
               target.append([word] + j)

    return target



if __name__ == '__main__':
    target = all_cut(sentence, Dict)
    print(target)
