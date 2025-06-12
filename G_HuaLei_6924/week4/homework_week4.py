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
sentence_to_cut = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    target = []
    recursive_cut(sentence, Dict, [], target)
    return target


def recursive_cut(sentence, Dict, cut_arr, target):
    # print(f'sentence: {sentence}')
    # print(f'cut_arr_1: {cut_arr}')
    # print(f'target: {target}\n')
    has_cut_all = False
    if sentence in Dict:
        has_cut_all = True
        cut_arr.append(sentence)
        target.append(cut_arr)

    for i in range(len(sentence)):
        if has_cut_all:
            cut_arr = cut_arr[:-1]
        char_seq = sentence[:i+1]
        if char_seq in Dict:
            # print(f'cut_arr_0: {cut_arr}')
            # print(f'char_seq: {char_seq}')
            recursive_cut(sentence[i+1:], Dict, [*cut_arr, char_seq], target)

res_target = all_cut(sentence_to_cut, Dict)

for index, item in enumerate(res_target):
    print(f'第{index+1}个：\n{item}')

#目标输出;顺序不重要
# target = [
#     ['经常', '有意见', '分歧'],
#     ['经常', '有意见', '分', '歧'],
#     ['经常', '有', '意见', '分歧'],
#     ['经常', '有', '意见', '分', '歧'],
#     ['经常', '有', '意', '见分歧'],
#     ['经常', '有', '意', '见', '分歧'],
#     ['经常', '有', '意', '见', '分', '歧'],
#     ['经', '常', '有意见', '分歧'],
#     ['经', '常', '有意见', '分', '歧'],
#     ['经', '常', '有', '意见', '分歧'],
#     ['经', '常', '有', '意见', '分', '歧'],
#     ['经', '常', '有', '意', '见分歧'],
#     ['经', '常', '有', '意', '见', '分歧'],
#     ['经', '常', '有', '意', '见', '分', '歧']
# ]

