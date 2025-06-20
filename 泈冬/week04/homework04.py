import jieba
# jieba.cut
#词典，每个词后方存储的是其词频，仅为示例，也可自行添加
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


#根据上方词典，对于输入文本，构造一个存储有所有切分方式的信息字典
#学术叫法为有向无环图，DAG（Directed Acyclic Graph），不理解也不用纠结，只当是个专属名词就好
#这段代码直接来自于jieba分词
# jieba.cut
def calc_dag(sentence):
        DAG = {}    #DAG空字典，用来存储DAG有向无环图
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N:
                if frag in Dict:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

sentence = "经常有意见分歧"
print(calc_dag(sentence))
#结果应该为{0: [0, 1], 1: [1], 2: [2, 4], 3: [3, 4], 4: [4, 6], 5: [5, 6], 6: [6]}
#0:[0,1]代表句子中的第0个字，可以单独成词，或与第1个字一起成词
#2:[2,4]代表句子中的第2个字，可以单独成词，或第2-4个字一起成词
#依次类推
#这个字典中实际上就存储了所有可能的切分方式的信息


#将DAG中的信息解码（还原）出来，用文本展示出所有切分方式
class DAGDecode:
    #通过两个队列来实现
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = calc_dag(sentence)  #使用了上方的函数
        self.length = len(sentence)
        self.unfinish_path = [[]]   #保存待解码序列的队列
        self.finish_path = []  #保存解码完成的序列的队列

    #对于每一个序列，检查是否需要继续解码
    #不需要继续解码的，放入解码完成队列
    #需要继续解码的，将生成的新队列，放入待解码队列
    #path形如:["经常", "有", "意见"]
    def decode_next(self, path):
        path_length = len("".join(path))
        if path_length == self.length:  #已完成解码
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.sentence[path_length:candidate+1]])
        self.unfinish_path += new_paths  #放入待解码对列
        return

    #递归调用序列解码过程
    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop() #从待解码队列中取出一个序列
            self.decode_next(path)     #使用该序列进行解码


sentence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(dd.finish_path)
