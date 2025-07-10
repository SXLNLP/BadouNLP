import jieba
import math

"""
DAG 分词器：实现两种分词方式
1. Full segmentation — 穷举所有切分路径
2. Best segmentation — 动态规划选取最优路径
"""

# 示例词典，包含词频
WORD_DICT = {
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
    "分": 0.1
}


def build_dag(sentence, word_dict):
    """
    构建有向无环图（DAG），记录每个起点可到达的终点索引列表。
    每个节点表示一个起始字符索引，每条边表示词典中存在的子串。
    """
    dag = {}  # 存储 DAG 结构
    n = len(sentence)
    for start in range(n):  # 遍历每个起点
        end_list = []  # 当前起点能到达的合法终点
        i = start
        frag = sentence[start]  # 从当前字符开始构造子串
        while i < n:
            if frag in word_dict:  # 如果子串存在于词典中
                end_list.append(i)  # 添加当前终点索引
            i += 1
            frag = sentence[start:i + 1]  # 扩展子串范围
        if not end_list:  # 如果没有合法词，按单字切分
            end_list.append(start)
        dag[start] = end_list  # 存入 DAG
    return dag


class DAGFullSegmenter:
    """
    穷举所有可能分词路径，使用 DFS 方式遍历 DAG。
    """

    def __init__(self, sentence, word_dict):
        self.sentence = sentence
        self.dag = build_dag(sentence, word_dict)  # 构建 DAG
        self.length = len(sentence)
        self.unfinished_paths = [[]]
        self.finished_paths = []

    def decode_next(self, path):
        """
        扩展路径：从当前路径末尾开始，根据 DAG 添加下一个词。
        """
        current_index = len("".join(path))  # 当前路径覆盖的字符长度
        if current_index == self.length:  # 如果到达句末
            self.finished_paths.append(path)  # 添加到完成路径
            return
        for end in self.dag[current_index]:  # 遍历所有合法下一个词的终点
            word = self.sentence[current_index:end + 1]  # 提取子串
            self.unfinished_paths.append(path + [word])  # 加入新路径

    def decode(self):
        """
        主解码函数，遍历所有可能路径。
        """
        while self.unfinished_paths:
            path = self.unfinished_paths.pop()  # 取出一条未完成路径
            self.decode_next(path)  # 尝试扩展路径
        return self.finished_paths


class DAGBestSegmenter:
    """
    选取概率最大的分词路径，使用动态规划实现。
    """

    def __init__(self, sentence, word_dict):
        self.sentence = sentence
        self.word_dict = word_dict
        self.dag = build_dag(sentence, word_dict)  # 构建 DAG
        self.route = {}  # 存储每个位置的最优路径
        self.length = len(sentence)

    def calc_route(self):
        """
        动态规划计算每个位置起点的最优路径概率。
        """
        self.route[self.length] = (0, 0)  # 句尾初始化，概率为0
        for idx in range(self.length - 1, -1, -1):  # 从后往前计算
            candidates = []  # 存储所有可能的路径选择
            for end in self.dag[idx]:  # 遍历所有合法的词终点
                word = self.sentence[idx:end + 1]  # 当前词
                prob = math.log(self.word_dict.get(word, 1e-6))  # 当前词概率的对数
                next_prob = self.route[end + 1][0]  # 从下一个位置起的最优路径概率
                candidates.append((prob + next_prob, end + 1))  # 总概率与下一个位置索引
            self.route[idx] = max(candidates)  # 选择概率最大的路径

    def decode(self):
        """
        使用预先计算的 route 解码出最优路径。
        """
        self.calc_route()  # 先计算最优路径
        idx = 0
        result = []
        while idx < self.length:  # 从头开始解码路径
            next_idx = self.route[idx][1]  # 下一词的终点
            result.append(self.sentence[idx:next_idx])  # 添加词到结果中
            idx = next_idx
        return result


if __name__ == '__main__':
    sentence = "经常有意见分歧"
    """
    经常 / 有意见 / 分歧
    经常 / 有意见 / 分 / 歧
    经常 / 有 / 意见 / 分歧
    经常 / 有 / 意见 / 分 / 歧
    经常 / 有 / 意 / 见分歧
    经常 / 有 / 意 / 见 / 分歧
    经常 / 有 / 意 / 见 / 分 / 歧
    经 / 常 / 有意见 / 分歧
    经 / 常 / 有意见 / 分 / 歧
    经 / 常 / 有 / 意见 / 分歧
    经 / 常 / 有 / 意见 / 分 / 歧
    经 / 常 / 有 / 意 / 见分歧
    经 / 常 / 有 / 意 / 见 / 分歧
    经 / 常 / 有 / 意 / 见 / 分 / 歧
    """

    print(f"原句：{sentence}\n")

    # 打印 DAG 图结构
    dag = build_dag(sentence, WORD_DICT)

    print("DAG 结构：")
    for start, ends in dag.items():
        print(f"  {start}: {ends}")
    print()

    # 穷举所有切分方式
    full_seg = DAGFullSegmenter(sentence, WORD_DICT)
    all_paths = full_seg.decode()

    print("【全路径穷举切分】")
    for path in all_paths:
        print("  " + " / ".join(path))
    print()

    # 最优路径切分
    best_seg = DAGBestSegmenter(sentence, WORD_DICT)
    best_paths = best_seg.decode()

    print("【最优路径切分】")
    print("  " + " / ".join(best_paths))
