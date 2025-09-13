"""
任务型多轮对话系统
读取场景脚本完成多轮对话

编程建议：
1.先搭建出整体框架
2.先写测试用例，确定使用方法
3.每个方法不要超过20行
4.变量命名和方法命名尽量标准
5.通过注释引导AI工具生成辅助代码
"""

import json
import pandas as pd
import re
import os
from collections import defaultdict


class DialogueSystem:
    def __init__(self):
        self.all_node_info = {}
        self.slot_templet = None
        self.slot_info = defaultdict(dict)     # 三列：slot, query, values
        self.last_response = None  # 存储上一次的回复
        self.last_memory_state = None  # 存储上一次的内存状态
        self.load()

    def load(self):
        self.load_scenario("scenario/scenario-买衣服.json")
        self.load_slot_templet("scenario/slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]  # scenario-买衣服
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "child_node" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['child_node'] = [scenario_name + "_" + x for x in
                                                                                      node['child_node']]

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        # 逐行读取，slot为key，query和values为value
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values

    def is_repeat_request(self, query):
        """
        判断用户是否请求重复上次回复
        """
        repeat_keywords = ['什么', '再说', '重复', '没听清', '没听懂', '没听到', '没明白', '没听明白',
                           '没听清楚', '再说一遍', '重复一遍', '再说一次', '重复一次']

        # 检查是否包含重复关键词
        for keyword in repeat_keywords:
            if keyword in query:
                return True
        return False

    def nlu(self, memory):
        # 自然语言理解
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        # 意图识别，匹配当前可以访问的节点
        query = memory['query']
        max_score = -1
        hit_node = None

        # 如果是重复请求，直接返回上次的节点
        if self.is_repeat_request(query) and self.last_memory_state:
            memory["hit_node"] = self.last_memory_state.get("hit_node")
            memory["intent_score"] = 1.0  # 设置为最高分
            memory["is_repeat"] = True  # 标记为重复请求
            return memory

        for node in memory["available_nodes"]:
            score = self.calculate_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        memory["is_repeat"] = False  # 标记为非重复请求
        return memory

    def calculate_node_score(self, query, node):
        # 节点意图打分，算和intent相似度
        intent = self.all_node_info[node]['intent']
        max_score = -1
        for sentence in intent:
            score = self.calculate_sentence_score(query, sentence)
            max_score = max(max_score, score)
        return max_score

    def calculate_sentence_score(self, query, sentence):
        # 两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)

    def slot_filling(self, memory):
        # 槽位填充
        # 如果是重复请求，不需要重新填充槽位
        if memory.get("is_repeat", False):
            return memory

        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, query):
                    memory[slot] = re.search(slot_values, query).group()
        return memory

    def dst(self, memory):
        # 如果是重复请求，保持原来的状态
        if memory.get("is_repeat", False) and self.last_memory_state:
            memory["require_slot"] = self.last_memory_state.get("require_slot")
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
        # 策略优化
        # 如果是重复请求，保持原来的策略
        if memory.get("is_repeat", False) and self.last_memory_state:
            memory["policy"] = self.last_memory_state.get("policy")
            memory["available_nodes"] = self.last_memory_state.get("available_nodes", [])
            return memory

        if memory["require_slot"] is None:
            # 没有需要填充的槽位
            memory["policy"] = "reply"
            # self.take_action(memory)
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("child_node", [])
        else:
            # 有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点
        return memory

    def nlg(self, memory):
        # 根据policy执行反问或回答
        # 如果是重复请求，直接返回上次的回复
        if memory.get("is_repeat", False) and self.last_response:
            memory["response"] = self.last_response
            return memory

        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            # policy == "request"
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]

        # 保存当前回复供下次可能使用
        self.last_response = memory["response"]
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            template = template.replace(slot, memory[slot])
        return template

    def run(self, query, memory):
        """
        query: 用户输入
        memory: 用户状态
        """
        memory["query"] = query

        # 保存上一次的内存状态（用于重复功能）
        self.last_memory_state = memory.copy()

        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    memory = {"available_nodes": ["scenario-买衣服_node1"]}  # 用户状态

    print("对话系统已启动！输入'什么'、'再说一遍'等可以重复上次回复。")
    print("===========")

    while True:
        query = input("用户输入：")
        if query.lower() in ['退出', 'exit', 'quit', 'q']:
            print("系统：再见！")
            break

        memory = ds.run(query, memory)
        response = memory['response']
        print(f"系统回复：{response}")
        print("===========")
