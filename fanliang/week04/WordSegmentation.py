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

def cut_do(targetlist,itemarray,index,sentence,Dict): #递归函数，用于生成所有可能的切分方式
    for iterm in Dict:
        if(sentence[index:].startswith(iterm)):
            curitemarray = itemarray.copy()
            curitemarray.append(iterm)
            if(index+len(iterm) == len(sentence)):
                targetlist.append(curitemarray)
            else:
                cut_do(targetlist,curitemarray,index+len(iterm),sentence,Dict)
    return
#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    targetlist = []
    itemarray = []
    cut_do(targetlist,itemarray,0,sentence,Dict)
    for item in targetlist:
        print(item)

all_cut(sentence,Dict)
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

