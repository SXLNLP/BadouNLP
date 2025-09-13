'''
先捋清任务
写注释：记录文件是干什么的
任务型多轮对话系统
读取场景脚本完成多轮对话

# 编程建议：
1、先搭出整体框架
2、先写测试用例，确定使用方法
3、每个方法不要超过20行，细分成非常小的子任务
4、变量命名和方法命名尽量标准
'''

import json
import pandas as pd
import re
import os


class DialogueSystem:

    def __init__(self):
        self.load()
        # 重听话术列表
        self.repeat_keywords = [
            "我没听清楚",
            "请你重复一遍",
            "可以重说一遍吗",
            "我没听明白"
        ]
        self.similarity_threshold = 0.5  # 相似度阈值

    def load(self):
        # 不同的场景将节点分开，记录一个全局变量
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服2.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_template("slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        # 读取json文件
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in
                                                                                     node['childnode']]

    def load_slot_template(self, file):
        # 读取excel文件
        self.slot_template = pd.read_excel(file)
        self.slot_info = {}
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
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calculate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calculate_sentence_score(self, query, sentence):
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        if len(union) == 0:  # 防止除以零
            return 0
        return len(intersection) / len(union)

    def slot_filling(self, memory):
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        query = memory['query']
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
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点
        return memory

    def nlg(self, memory):
        # 根据policy执行反问或者问答
        if memory["policy"] == 'reply':
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        elif memory["policy"] == "request":
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]
        elif memory["policy"] == "repeat":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)

        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            template = template.replace(slot, memory.get(slot, ""))
        return template

    def handle_repeat_request(self, memory):
        # 设置policy为repeat
        memory["policy"] = "repeat"
        return memory

    def is_repeat_request(self, query):
        for repeat_sentence in self.repeat_keywords:
            score = self.calculate_sentence_score(query, repeat_sentence)
            if score >= self.similarity_threshold:
                return True
        return False

    def run(self, query, memory):
        '''
        query: 用户输入
        memory： 用户状态
        :param query:
        :param memory:
        :return:
        '''
        memory["query"] = query

        # 检测用户是否请求重听
        if self.is_repeat_request(query):
            return self.handle_repeat_request(memory)

        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    memory = {"available_nodes": ["scenario-买衣服2_node1", "scenario-看电影_node1"]}  # 用户状态
    while True:
        query = input("请输入：")
        memory = ds.run(query, memory)
        print(memory)
        response = memory['response']
        print(response)
        print("=============")
