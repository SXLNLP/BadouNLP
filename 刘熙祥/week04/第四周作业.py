#week3作业
import jieba
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

    if sentence is None or len(sentence) == 0:
        return [[]]
    target = []
    for i in range(len(sentence)):
        if sentence[:i+1] in Dict:
            print("第%d次: " % (i + 1) + sentence[:i+1] +","+ sentence[i+1:])
            for j in all_cut(sentence[i+1:], Dict):
                target.append([sentence[:i+1]]+j)
                print("%d, %s, %s, %s" %(i+1, target, [sentence[:i+1]], j))
    return target

def main():
    print(all_cut(sentence, Dict))

if __name__ == "__main__":
    main()

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

# String = "今儿天气不错，适合出去游玩"
# jieba.add_word("天气不错")
# print(list(jieba.cut(String)))
