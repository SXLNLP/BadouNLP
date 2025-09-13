'''
任务型多轮对话系统
读取场景脚本完成多轮对话，包含重听功能和重复购买支持
'''

import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
        # 存储主节点ID，用于支持流程结束后重新开始
        self.main_node_id = None

    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario/scenario-买衣服.json")
        self.load_slot_templet("scenario/slot_fitting_templet.xlsx")


    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        # 记录主节点ID（第一个节点）
        if scenario:
            self.main_node_id = scenario_name + "_" + scenario[0]['id']

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

    def nlu(self, memory):
        # 先检查是否是重听请求
        if self.is_repeat_request(memory['query']):
            memory['is_repeat'] = True
            return memory

        memory['is_repeat'] = False
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def is_repeat_request(self, query):
        """判断用户是否请求重听上一条回复"""
        repeat_keywords = ['重听', '再说一遍', '重复', '刚才说什么', '没听清', '在说一遍']
        query_lower = query.lower()
        for keyword in repeat_keywords:
            if keyword in query_lower:
                return True
        return False

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

        # 特殊处理：如果没有可用节点但检测到购买意图，允许重新开始
        if hit_node is None and self.main_node_id and "买" in query and "衣服" in query:
            hit_node = self.main_node_id
            max_score = 0.5  # 给予一个中等分数

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
        return len(intersection) / len(union) if union else 0


    def slot_filling(self, memory):
        #槽位填充
        hit_node = memory["hit_node"]
        if not hit_node:  # 检查hit_node是否为None
            return memory

        query = memory['query']
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                # 分割可能的多个值选项
                possible_values = slot_values.split('|')
                for value in possible_values:
                    if value in query:
                        memory[slot] = value
                        break
        return memory

    def dst(self, memory):
        # 如果是重听请求，不需要处理槽位
        if memory.get('is_repeat', False):
            return memory

        hit_node = memory["hit_node"]
        if not hit_node:  # 检查hit_node是否为None
            memory["require_slot"] = None
            return memory

        # 如果是从主节点重新开始，清除之前的槽位信息
        if hit_node == self.main_node_id:
            slots_to_clear = [slot for slot in memory if slot.startswith('#')]
            for slot in slots_to_clear:
                del memory[slot]

        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        # 如果是重听请求，不需要改变策略
        if memory.get('is_repeat', False):
            return memory

        # 检查hit_node是否为None
        if memory["hit_node"] is None:
            memory["policy"] = "unrecognized"
            return memory

        if memory["require_slot"] is None:
            #没有需要填充的槽位
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            #有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]] #停留在当前节点
        return memory

    def nlg(self, memory):
        # 如果是重听请求，返回上一条回复
        if memory.get('is_repeat', False):
            if 'last_response' in memory:
                memory["response"] = memory['last_response']
            else:
                memory["response"] = "抱歉，我还没有说过任何内容"
            return memory

        # 处理无法识别意图的情况
        if memory.get("policy") == "unrecognized":
            memory["response"] = "抱歉，我没有理解您的意思，请换一种说法"
            memory['last_response'] = memory["response"]
            return memory

        #根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            #policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]

        # 保存当前回复作为下一次可能的重听内容
        memory['last_response'] = memory["response"]
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
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    print("节点信息:", ds.all_node_info)
    print("槽位信息:", ds.slot_info)

    memory = {
        "available_nodes": ["scenario-买衣服_node1"] if ds.main_node_id is None else [ds.main_node_id],
        "last_response": ""  # 用于存储上一条回复，支持重听功能
    }  # 用户状态

    print("\n===== 服装购买对话系统 =====")
    print("提示：您可以随时输入'重听'、'再说一遍'等请求重复上一条回复")
    print("提示：输入'exit'、'quit'或'退出'可结束对话")
    while True:
        query = input("请输入：")
        if query.lower() in ['exit', 'quit', '退出']:
            print("对话结束，谢谢使用！")
            break

        memory = ds.run(query, memory)
        response = memory['response']
        print("回复：", response)
        print("==========================")
