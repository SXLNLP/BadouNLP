
Dict = {
    "计算机": 0.1,
    "网络": 0.1,
    "编程": 0.1,
    "语言": 0.1,
    "编程语言": 0.2,
    "开发": 0.2,
    "软件": 0.2,
    "工程": 0.1,
    "软件工程": 0.3,
    "系统": 0.1,
    "操作系统": 0.3,
    "操作": 0.1,
    "数据": 0.2,
    "数据库": 0.3,
    "库": 0.1,
    "是": 0.05,
    "的": 0.05,
    "和": 0.05,
    "基础": 0.1
}

# 待切分的 IT 文章片段
sentence = "软件工程是开发操作系统和数据库的基础"

def all_cut(sentence, Dict):
    """
    实现全切分：给定一个句子，返回所有可能的切分方式，
    每个切分方式中的词都必须存在于词典 Dict 中。
    使用递归 + 备忘录方式加速。
    """
    memo = {}  # 用于存储已计算过的子句切分结果

    def cut(sub_sentence):
        # 如果该子句已经计算过，则直接返回缓存结果
        if sub_sentence in memo:
            return memo[sub_sentence]

        # 空字符串表示完成切分，返回一个空列表作为有效路径
        if not sub_sentence:
            return [[]]

        result = []

        # 遍历所有前缀
        for i in range(1, len(sub_sentence) + 1):
            prefix = sub_sentence[:i]
            if prefix in Dict:
                # 前缀合法，递归切分剩下部分
                suffix_cuts = cut(sub_sentence[i:])
                for cut_result in suffix_cuts:
                    result.append([prefix] + cut_result)

        memo[sub_sentence] = result
        return result

    return cut(sentence)

# 主程序运行测试
if __name__ == "__main__":
    print(f"待切分语句：{sentence}\n\n所有可行切分方式如下：")
    results = all_cut(sentence, Dict)
    for idx, res in enumerate(results, 1):
        print(f"{idx}. {' | '.join(res)}")
