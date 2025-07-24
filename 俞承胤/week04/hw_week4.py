import copy
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
        "分":0.1,
        # "我": 0.05,
        # "是": 0.05,
        # "值": 0.05,
        # "日": 0.05,
        # "生": 0.05,
        # "值日": 0.05,
        # "值日生": 0.05,
        # "有毛病": 0.05,
        # "也": 0.05,
        # "想": 0.05,
        # "过": 0.05,
        # "过过": 0.05,
        # "过儿": 0.05,
        # "儿": 0.05,
        # "的": 0.05,
        # "日": 0.05,
        # "子": 0.05,
        # "日子": 0.05
        }

#待切分文本
sentence = "经常有意见分歧"
# sentence = "我是值日生"
# sentence = "我也想过过过儿过过的日子"
#实现全切分函数，输出根据字典能够切分出的所有的切分方式


def all_cut(sentence, Dict):
    i = 0
    result_list = []
    for s in sentence:
        print("文字",s)
        d_list = [d for d in Dict if d.startswith(s)]
        print("关联到的分词",d_list)
        #当前已整理的排列集合
        list = copy.deepcopy(result_list)
        #当前已整理的排列集合，为了获取当前分词无法匹配的排列
        oragen_list = copy.deepcopy(result_list)
        result_list = []
        for d in d_list:
            print("处理的分词", d)
            if i == 0:
                if sentence.startswith(d):
                    result_list.append([d])
                    print(f"首次获取直接放入列表", result_list)
            else:
                end_list = [l for l in list if sentence.startswith("".join(l) + d)]
                print("与分词匹配的句子", len(end_list), end_list)
                #深拷贝为了值的内容不受影响
                temp_list = copy.deepcopy(end_list)
                # 匹配的句子根据分词数量成倍增加
                for t in temp_list:
                    t.append(d)
                    print("t", t, d)
                result_list.extend(temp_list)
                print(f"{s}：加temp列表", result_list)
                # 匹配列表空代表无需添加此分词，无匹配不了的排序，也不需要添加
                if len(end_list) > 0 and len(oragen_list) > 0:
                    # 每个同开头的分词，从同一个集合中拿走可匹配的排列，拿不走的，保留在列表中
                    for e in end_list:
                        if e in oragen_list:
                            oragen_list.remove(e)
                    print("与分词无关的句子", len(oragen_list),oragen_list)
        # 所有同开头分词处理完后才添加到队列
        if len(oragen_list) > 0:
            result_list.extend(oragen_list)
            print(f"{s}：加oragen列表", result_list)
        i = i + 1
        print(f"{s}：列表", result_list)

    #return target


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

if __name__ == "__main__":
    print(all_cut(sentence, Dict))

