"""
任务型多轮对话系统
读取场景脚本完成多轮对话

编程建议：
1.先搭建出整体框架
2.先写测试用例，确定使用方法
3.每个方法不要超过20行
4.变量命名和方法命名尽量标准
"""

import json
import pandas as pd
import re
import os
import random

class DialogueSystem:
    def __init__(self):
        self.all_node_info = None
        self.slot_template = None
        self.slot_info = None
        self.load()
    
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
        self.slot_template = pd.read_excel(file)
        #三列：slot, query, values
        self.slot_info = {}
        #逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_template)):
            slot = self.slot_template.iloc[i]['slot']
            query = self.slot_template.iloc[i]['query']
            values = self.slot_template.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values
      
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
            score = self.calculate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory


    def calculate_node_score(self, query, node):
        #节点意图打分，算和intent相似度
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calculate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score
    
    @staticmethod
    def calculate_sentence_score(query, sentence):
        #两个字符串做文本相似度计算。距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)


    def slot_filling(self, memory):
        #槽位填充
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        query = memory['query']  # 从memory中获取query
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
        """
        随机对话机制：随机决定听不清楚还是据实回答
        query: 用户输入
        memory: 用户状态
        """
        # 检查当前是否在等待用户重复
        if memory.get("waiting_for_repeat", False):
            # 第二轮：处理之前暂存的查询
            actual_query = memory.get("pending_query", "")
            memory["query"] = actual_query
            
            # 重置双轮状态
            memory["waiting_for_repeat"] = False
            memory["pending_query"] = None
            
            # 执行正常对话流程处理暂存的查询
            memory = self.nlu(memory)
            memory = self.dst(memory)
            memory = self.dpo(memory)
            memory = self.nlg(memory)
            return memory
        else:
            # 随机决策：50%概率听不清楚，50%概率据实回答
            hear_clearly = random.choice([True, False])
            
            if hear_clearly:
                # 据实回答：直接执行正常对话流程
                memory["query"] = query
                memory = self.nlu(memory)
                memory = self.dst(memory)
                memory = self.dpo(memory)
                memory = self.nlg(memory)
                # 记录此次选择了据实回答
                memory["last_decision"] = "heard_clearly"
                return memory
            else:
                # 听不清楚：暂存查询并要求重复
                memory["pending_query"] = query
                memory["waiting_for_repeat"] = True
                memory["response"] = "你好，我听不清楚你说的，请再说一遍"
                # 记录此次选择了听不清楚
                memory["last_decision"] = "cant_hear"
                return memory
        

if __name__ == '__main__':
    ds = DialogueSystem()
    print(ds.slot_info)

    memory = {"available_nodes":["scenario-买衣服_node1"]} #用户状态
    while True:
        query = input("请输入：")
        # query = "你好"    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
        





