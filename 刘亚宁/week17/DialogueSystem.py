import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
        self.last_response = None  # 存储上一次的回复
    
    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
                
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in node['childnode']]
            

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        # 三列：slot, query, values
        self.slot_info = {}
        # 逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values
      
    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        # 意图识别，匹配当前可以访问的节点
        query = memory['query']
        
        # 检查是否是重听请求
        if self.is_repeat_request(query):
            memory["is_repeat_request"] = True
            return memory
            
        memory["is_repeat_request"] = False
        
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

    def is_repeat_request(self, query):
        # 判断是否是重听请求
        repeat_patterns = [
            "重复", "再说一遍", "没听清", "没听懂", "什么", "你说什么", 
            "repeat", "again", "pardon", "没听清楚"
        ]
        for pattern in repeat_patterns:
            if pattern in query:
                return True
        return False

    def calucate_node_score(self, query, node):
        # 节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    def calucate_sentence_score(self, query, sentence):
        # 两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)

    def slot_filling(self, memory):
        # 槽位填充
        if memory["is_repeat_request"]:
            return memory
            
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, memory['query']):
                    memory[slot] = re.search(slot_values, memory['query']).group()
        return memory

    def dst(self, memory):
        if memory["is_repeat_request"]:
            return memory
            
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
        if memory["is_repeat_request"]:
            memory["policy"] = "repeat"
            return memory
            
        if memory["require_slot"] is None:
            # 没有需要填充的槽位
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            # 有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]] # 停留在当前节点
        return memory

    def nlg(self, memory):
        # 根据policy执行反问或回答
        if memory["policy"] == "repeat":
            # 重听功能：返回上一次的回复
            if self.last_response:
                memory["response"] = f"（重复）{self.last_response}"
            else:
                memory["response"] = "抱歉，没有之前的对话记录。"
        elif memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
            # 保存当前回复，以便后续重听
            self.last_response = memory["response"]
        else:
            # policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
            # 保存当前回复，以便后续重听
            self.last_response = memory["response"]
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            if slot in memory:
                template = template.replace(slot, memory[slot])
        return template

    def run(self, query, memory):
        '''
        query: 用户输入
        memory: 用户状态
        '''
        memory["query"] = query
        memory = self.nlu(memory)
        if not memory["is_repeat_request"]:
            memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory
        

if __name__ == '__main__':
    ds = DialogueSystem()
    print(ds.slot_info)

    memory = {"available_nodes":["scenario-买衣服_node1"]} # 用户状态
    print("对话系统已启动，输入'退出'结束对话")
    
    while True:
        query = input("用户输入：")
        
        if query in ["退出", "结束", "exit", "quit"]:
            print("对话结束，再见！")
            break
    
        memory = ds.run(query, memory)
        print(f"系统回复：{memory['response']}")
        print("=" * 50)
