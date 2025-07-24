import jieba

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

def calc_dag(sentence):
    DAG = {}
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

class DAGDecode:
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = calc_dag(sentence)
        self.length = len(sentence)
        self.unfinish_path = [[]]
        self.finish_path = []

    def decode_next(self, path):
        path_length = len("".join(path))
        if path_length == self.length:
            self.finish_path.append(path)
            return
        candidates = self.DAG[path_length]
        new_paths = []
        for candidate in candidates:
            new_paths.append(path + [self.sentence[path_length:candidate+1]])
        self.unfinish_path += new_paths
        return

    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop()
            self.decode_next(path)

def all_cut(sentence, Dict):
    dd = DAGDecode(sentence)
    dd.decode()
    return dd.finish_path

sentence = "经常有意见分歧"
target = all_cut(sentence, Dict)
print(target)
