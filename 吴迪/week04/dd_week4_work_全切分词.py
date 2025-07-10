# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {
    "经常": 0.1,
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
    "分": 0.1,
}


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence):
    DAG = {}  # 初始化DAG字典用于存储所有可能的切分方式
    N = len(sentence)  # 获取句子长度

    # 遍历每个字的长度
    for k in range(N):
        tmplist = []  # 保存当前位置 k 所有可能的结束位置
        i = k  # 从当前位置 k 开始
        frag = sentence[k]  # frag初始化为当前字符sentence[k]

        # 向后查找可能的词语
        while i < N:
            if frag in Dict:  # 在词中找到这个词
                tmplist.append(i)  # 记录结束位置
            i += 1  # 继续查询后面新词
            frag = sentence[k : i + 1]  # 更新 frag 为扩展后的新词 sentence[k:i+1]

        if not tmplist:  # 如果没有找到词
            tmplist.append(k)  # 如果没有匹配，强制加入词

        DAG[k] = tmplist  # 将 k 位置的所有可能结束位置，存入DAG
    return DAG


# 待切分文本
sentence = "经常有意见分歧"
print(f"====检查DAG结构====", all_cut(sentence))


# DAG中的信息解码出来，用文本展示所有切分方式
class DAGDecode:
    def __init__(self, sentence):
        self.sentence = sentence
        self.DAG = all_cut(sentence)  # 获取DAG
        self.length = len(sentence)
        self.unfinish_path = [[]]  # 保存代解析吗的序列的队列
        self.finish_path = []  # 保存解码完成的序列的队列

    """ 每一个序列检查是否需要继续解码，不需要继续解码的放入finish_path解码完成的队列，需要解码的
        放入待解码unfinish_path 队列
    """

    def decode_next(self, path):
        path_length = len("".join(path))  # 计算当前路径已经覆盖的长度

        if path_length == self.length:  # 如果当前路径已经覆盖长度，加入完整队列
            self.finish_path.append(path)
            return

        # 获取当前位置所有可能的结束位置
        candidates = self.DAG[path_length]

        # 取出新词，从当前位置path_length到conditate+1
        new_paths = []
        for canditate in candidates:
            new_word = self.sentence[path_length : canditate + 1]  # 取出新词
            new_paths.append(path + [new_word])

        self.unfinish_path += new_paths#新路径添加到待处理队列
        return

    # 递归调用序列解码过程
    def decode(self):
        while self.unfinish_path != []:
            path = self.unfinish_path.pop()  # 从待解码队列中取出一个序列
            self.decode_next(path)  # 使用该序列进行解码


# 待切分文本
sentence = "经常有意见分歧"
dd = DAGDecode(sentence)
dd.decode()
print(f"===输出解码完成的队列====", dd.finish_path)

# 目标输出;顺序不重要
target = [
    ["经常", "有意见", "分歧"],
    ["经常", "有意见", "分", "歧"],
    ["经常", "有", "意见", "分歧"],
    ["经常", "有", "意见", "分", "歧"],
    ["经常", "有", "意", "见分歧"],
    ["经常", "有", "意", "见", "分歧"],
    ["经常", "有", "意", "见", "分", "歧"],
    ["经", "常", "有意见", "分歧"],
    ["经", "常", "有意见", "分", "歧"],
    ["经", "常", "有", "意见", "分歧"],
    ["经", "常", "有", "意见", "分", "歧"],
    ["经", "常", "有", "意", "见分歧"],
    ["经", "常", "有", "意", "见", "分歧"],
    ["经", "常", "有", "意", "见", "分", "歧"],
]
