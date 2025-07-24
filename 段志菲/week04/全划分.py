#week3作业
import json

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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

#待切分文本
sentence = "经常有意见分歧"


#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    max_len = max(len(w) for w in Dict.keys()) if Dict else 1
    def cutback(start_index, path, target):
        if start_index == len(sentence):
            #print(path)
            target.append(path.copy())
            return
        for end_index in range(start_index + 1, min(start_index + max_len, len(sentence)) + 1):  #len(sentence)+1太长递归层数太深，优化为词表中最长词长度
            word = sentence[start_index:end_index]
            #print(word)
            if word in Dict:
                path.append(word)
                #start_index += len(word) #不需要手动刷新start_index
                cutback(end_index, path, target)
                path.pop()

    target = []
    cutback(0, [], target)
    return target


#目标输出;顺序不重要
'''
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
'''


def main():
    target = all_cut(sentence, Dict)
    for i, cut in enumerate(target, 1):
        print(f"{i}. {cut}")


if __name__ == "__main__":
    main()

'''
实现思路：
循环递归实现
从头开始，一次取i个字符与词表进行比较，有的话进行切分，然后移动窗口，再次取i个字符与词表进行比较，有的话进行切分，然后移动窗口
'''
