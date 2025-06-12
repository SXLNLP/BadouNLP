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
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    dag = {}
    #step1 遍历每个节点获得一张 DAG（通过邻接表来表示）
    for i,char in enumerate(sentence):
        path = {}
        j = 1
        while(1):
            # 从 sentence中提取从 i 到i + j - 1 的字符，检查该字符是否在 Dict中存在
            path_cand = sentence[i:i+j]    # 提取了从 i 到i+j-1的字符了
            if path_cand in Dict:
                path[path_cand] = Dict[path_cand]            
            j = j + 1
            if i+j-1>len(sentence)-1:
                # 此时索引已溢出，说明已经找到了从 i 开始的全部合法路径
                break
        dag[char] = path
    print(dag)
    #step2  通过DFS算法遍历 DAG 输出 target 
    target=[]
    start_index = 0
    end_index = len(sentence)
    def dfs(index, current_path):
        if index == end_index:
            # 到达终点时也就是合适的时机，此时需要将当前路径添加到target中
            target.append(current_path)
            return
        
        all_path_available = dag[sentence[index]]
        for path_available in all_path_available:
            index_next = index + len(path_available) if index + len(path_available) <= end_index else None
            if index_next:
              # 对于每一个路径分叉点，都深拷贝一个新的路径，使得新路径彼此之间的修改互不影响
              # 每个分叉点都会在合适的时机被添加到target中
              new_path = current_path+[path_available] # 这里需要深拷贝，否则会修改原路径
              dfs(index_next, new_path)
    dfs(start_index, [])
   
    return target

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
    target=all_cut(sentence, Dict)
    print(target)
    print(len(target))
