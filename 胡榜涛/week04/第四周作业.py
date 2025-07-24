#week3作业
import copy
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

#设计一个回溯函数，使得它能够遍历所有分支，即所有可能得组合
def backtrack(sentence,vocab,start,path_tmp,target):
    if start == len(sentence):#设置退出条件，当索引的起始位置超出了句子的最后一位，代表到头了
        target.append(path_tmp.copy())#将当前分支存下来，这里也可以使用deepcopy(),只是单层列表，没区别
        return #退出本函数，这个尤为重要，递归函数要设置退出条件，不然就无线递归了
    for end in range(start+1,len(sentence)+1):#为什么是len(sentence)+1呢，是因为range是左闭右开的，只会产生右边界前一位，本来[]索引就是右开的，所以就需要加一了
        word=sentence[start:end]
        if word in vocab:
            path_tmp.append(word)
            backtrack(sentence,vocab,end,path_tmp,target)
            path_tmp.pop() #存好一个可能分支后，就弹出最后一个，回退一下，这是点睛之笔

def all_cut(sentence,vocab):
    target=[]
    start=0
    path_tmp=[]
    backtrack(sentence,vocab,start,path_tmp,target)
    return target

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

if __name__ == '__main__':
    target_pre=all_cut(sentence,list(Dict.keys()))
    for word in target_pre:
        print(word)
