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
class Solution:
    def __init__(self,Dict):
        self.res = []
        self.list_word = []
        self.Dict = Dict
    def back(self, sentencce: str, sen_len: int,max_len: int):
        if sen_len == len(sentence):
            self.res.append(self.list_word.copy())
            return
        for i in range(sen_len+1, min(sen_len+max_len+1, len(sentence)+1)):
            word = sentencce[sen_len:i]
            if word in self.Dict:
                self.list_word.append(word)
                self.back(sentence,i,max_len)
                self.list_word.pop()
    #实现全切分函数，输出根据字典能够切分出的所有的切分方式
    def all_cut(self,sentence):
        max_len=0
        for word in self.Dict.keys():
            max_len = max(max_len,len(word))
        self.back(sentence,0,max_len)
    # 返回最大词频的列表
    def max_count(self,sentence):
        self.all_cut(sentence)
        self.res.sort(key=lambda x:sum(self.Dict.get(word,0) for word in x),reverse=True)
        return self.res[0]
s = Solution(Dict)
print(s.max_count(sentence))


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
