# -*- coding: utf-8 -*-
#week4作业 - control
import itertools

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

# [经常，经，常，有，有意见，歧，意见，分歧，见，意，见分歧，分] -> 3
#  经常有意见分歧  <=> 最大左匹配
#  经常有/意见分歧  没有
#  经常 有意见分歧
#  2 * 3 * 2 = 12
#  [经常] [有意见] [分歧]  <=> 正向最大左  3
#  [经常] [有，意见]  [分歧]             2
#  [经，常] [有，意，见] [分， 歧]        1

#  2 * 1 * 1 * 3
#  [经常] [有] [意] [见分歧] <=> 逆向最大右 3
#  [经常] [有] [意] [见，分歧]            2
#  [经，常] [有] [意] [见，分，歧]
# =>> 排重
# max  ->   1
#

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    max_len = calcute_max_len(Dict)
    left_cut_result = all_left_cut(max_len, sentence, Dict)
    right_cut_result = all_right_cut(max_len, sentence, Dict)
    all_result = set(left_cut_result + right_cut_result)
    return [x.split(' ') for x in all_result]

# 所有最大左匹配
def all_left_cut(local_max_len, local_sentence, local_dict):
    #计算最大匹配,顶层最大匹配结果
    parent_left_cut = do_max_left_cut(local_max_len, local_sentence, local_dict)
    #构建二维数组，存储结果
    all_left = []
    for i in  range(len(parent_left_cut)):
        child = parent_left_cut[i]
        child_set = {child}
        all_left.append(child_set)
        for j in range(1,local_max_len):
            child_dict,max_dict_len = dict_filter(Dict, j)
            split_result = do_max_left_cut(j, child, child_dict)
            all_left[i].add(' '.join(split_result))
    finalResult = cartesian_product(all_left)
    return [' '.join(x) for x in finalResult]

def all_right_cut(local_max_len, local_sentence, local_dict):
    #计算最大匹配,顶层最大匹配结果
    parent_right_cut = do_max_right_cut(local_max_len, local_sentence, local_dict)
    #构建二维数组，存储结果
    all_right = []
    for i in  range(len(parent_right_cut)):
        child = parent_right_cut[i]
        child_set = {child}
        all_right.append(child_set)
        for j in range(1,local_max_len):
            child_dict,max_dict_len = dict_filter(Dict, j)
            split_result = do_max_right_cut(j, child, child_dict)

            all_right[i].add(' '.join(split_result))
    finalResult = cartesian_product(all_right)
    return [' '.join(x) for x in finalResult]

def cartesian_product(lists):
    match len(lists):
        case 2:
            return list(itertools.product(lists[0], lists[1]))
        case 3:
            return list(itertools.product(lists[0],lists[1],lists[2]))
        case 4:
            return list(itertools.product(lists[0], lists[1], lists[2], lists[3]))
        case _:
            return []


def dict_filter(Dict, max_len):
    new_dict = []
    for _, e in enumerate(Dict):
        if(len(e) <= max_len):
            new_dict.append(e)
    return new_dict,max_len

def calcute_max_len(Dict):
    max_len = 0
    for key in Dict.keys():
        max_len = max(max_len, len(key))
    return max_len

# 正向最大左匹配
def do_max_left_cut(max_len, local_sentence, dict):
    words = []
    start_index,end_index = 0, max_len
    while start_index < len(local_sentence):
        if start_index < end_index:
            subword = local_sentence[start_index:end_index]
            if subword in dict or len(subword) == 1:
                words.append(subword)
                start_index += len(subword)
                end_index = start_index + max_len
            else:
                end_index -= 1
        else:
            end_index += max_len
    return words

# 逆向最大右匹配
def do_max_right_cut(max_len, local_sentence, dict):
    words = []
    start_index,end_index = len(local_sentence) - max_len, len(local_sentence)
    while end_index > 0:
        if start_index < end_index:
            subword = local_sentence[start_index:end_index]
            if subword in dict or len(subword) == 1:
                words.append(subword)
                end_index -= len(subword)
                start_index = max(end_index - max_len, 0)
            else:
                start_index += 1
        else:
            end_index -= max_len
    words.reverse()
    return words


if __name__ == '__main__':
    result = all_cut(sentence, Dict)
    print(result)

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

