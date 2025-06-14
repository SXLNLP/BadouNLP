#week3作业
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

def generate_sets(min, max, target_sum):
    results = set()
    for val in range(1, target_sum+1):
        for combo in itertools.product(range(min, max+1), repeat=val):
            if sum(combo) == target_sum:
                results.add(combo)
    return results

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    max_window = 0
    min_window = 1
    for word in Dict:
        word_len = len(word)
        if word_len > max_window:
            max_window = word_len
    print("min_word_len: ", min_window)
    print("max_word_len: ", max_window)
    sentence_len = len(sentence)
    print("sentence_len: ", sentence_len)
    all_permutations = generate_sets(min_window, max_window, sentence_len)
    print(all_permutations)
    # print("sentence_len: ", sentence_len)
    #
    # win_sizes = []
    # for win_size in range(min_window, max_window+1):
    #     win_sizes.append(win_size)
    # print(win_sizes)
    # # cases 331/3211/31111/211111/1111111
    # multisets = [
    #     [3, 3, 1],
    #     [3, 2, 1, 1],
    #     [3, 1, 1, 1, 1],
    #     [2, 1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1, 1, 1]
    # ]
    # case1 = [3,3,1]
    # case2 = [3,2,1,1]
    # case3 = [3,1,1,1,1]
    # case4 = [2,1,1,1,1,1]
    # case5 = [1,1,1,1,1,1,1]
    #
    # all_permutations = []
    # for multiset in multisets:
    #     # 生成所有排列并去重（因有重复数字）
    #     permutations_set = set(itertools.permutations(multiset))
    #     all_permutations.extend(permutations_set)
    #
    # print(all_permutations)

    # all_permutations = []

    cut_list = []
    for case in all_permutations:
        start_index = 0
        cut = []
        for index in case:
            # print("index: ", index)
            end_index = start_index + index
            # print("start_index:", start_index)
            # print("end_index:", end_index)
            cut.append(sentence[start_index:end_index])
            start_index = end_index
        cut_list.append(cut)

    # print(cut_list)
    print(len(cut_list))
    target = []
    for cut in cut_list:
        target_flag = True
        for word in cut:
            if word not in Dict:
                target_flag = False
        if target_flag:
            target.append(cut)

    # print(target)
    print(len(target))

    return target

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

targets = all_cut(sentence, Dict)
for target in targets:
    print(target)
# print(generate_sets(1, 3, 7))
