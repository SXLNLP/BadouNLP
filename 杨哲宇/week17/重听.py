import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
    
    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        #self.again_que={} #创建一个空的节点，用于记录重听内容
        self.init_again_que()
        self.load_slot_templet("slot_fitting_templet.xlsx")

    def init_again_que(self):
        # 定义重复节点的基本信息
        again_que = {
            "id": "special_repeat_node",
            "intent": ["你说啥", "再说一遍"],
            "childnode": []  # 初始化子节点列表
             }
        # 将新节点添加到节点集合
        self.all_node_info[repeat_node["id"]] = again_que

        # 为每个现有节点添加重复节点作为子节点
        again-que_id = repeat_node["id"]
        for node in self.all_node_info.values():
            # 确保子节点列表存在，不存在则创建空列表
            if "childnode" not in node:
                node["childnode"] = []


    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:    #打开场景，加载变量
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]  #提取文件名
        #遍历节点
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in node['childnode']]
            

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        #三列：slot, query, values
        self.slot_info = {}
        #逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values
    #语义解析  
    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        #意图识别，匹配当前可以访问的节点
        query = memory['query']
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory


    def calucate_node_score(self, query, node):
        #节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    def calucate_sentence_score(self, query, sentence):
        #两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)


    def slot_filling(self, memory):
        #槽位填充
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, query):
                    memory[slot] = re.search(slot_values, query).group()
        return memory

    def dst(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        if memory["require_slot"] is None:
            #没有需要填充的槽位
            memory["policy"] = "reply"
            # self.take_action(memory)
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            #有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]] #停留在当前节点
        return memory

    def nlg(self, memory):
        #根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            #policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            template = template.replace(slot, memory[slot])
        return template

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

    def pm(self, memory):
        """处理对话策略，根据当前对话状态返回相应策略"""
        # 检查是否需要询问用户补充信息
        if memory.get("require_slot") is not None:
            strategy = {
                "available_node": [memory["hit_node"]],
                "policy": "ask"
            }
        # 检查是否需要重复上一次回复
        elif memory.get("state") == "repeat":
            strategy = {"policy": "repeat"}
        # 默认执行回答策略并更新可用节点
        else:
            current_node = self.all_node_info[memory["hit_node"]]
            strategy = {
                "available_node": current_node.get("childnode", []),
                "policy": "answer"
            }

        # 将策略更新到memory中
        memory.update(strategy)
        return memory

if __name__ == '__main__':
    ds = DialogueSystem()
    # print(ds.all_node_info)
    print(ds.slot_info)

    memory = {"available_nodes":["scenario-买衣服_node1","scenario-看电影_node1"]} #用户状态
    while True:
        query = input("请输入：")
        # query = "你好"    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
