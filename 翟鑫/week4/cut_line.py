# week3作业
import copy

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
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
        "分": 0.1,
        "是": 0.2,
        }

# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence):
    res = []
    do_all_cut(sentence, 0, res, [])
    return res


def do_all_cut(sentence: str, idx, res: list, cur_res: list):
    if idx >= len(sentence):
        res.append(copy.deepcopy(cur_res))
        return

    for i in range(idx + 1, len(sentence) + 1):
        w = sentence[idx: i]
        if w in Dict.keys():
            cur_res.append(w)
            do_all_cut(sentence, i, res, cur_res)
            cur_res.pop(-1)


if __name__ == '__main__':
    sentence = "经常有意见分歧"
    # sentence = "意见分歧是经常"
    res = all_cut(sentence)
    print(len(res), res)
