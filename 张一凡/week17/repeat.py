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
        self.load_slot_templet("slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario:
            self.all_node_info[scenario_name + "_" + node['id']] = node
            if "childnode" in node:
                self.all_node_info[scenario_name + "_" + node['id']]['childnode'] = [scenario_name + "_" + x for x in
                                                                                     node['childnode']]

    def load_slot_templet(self, file):
        self.slot_templet = pd.read_excel(file)
        self.slot_info = {}
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values

    def is_repeat_request(self, query):
        """判断是否为重听请求"""
        repeat_patterns = ["重复", "再说一遍", "重听", "没听清", "没听懂"]
        for pattern in repeat_patterns:
            if pattern in query:
                return True
        return False

    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        query = memory['query']

        # 先检查是否为重听请求
        if self.is_repeat_request(query):
            memory["is_repeat"] = True
            return memory

        memory["is_repeat"] = False
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
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calucate_sentence_score(query, sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calucate_sentence_score(self, query, sentence):
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)

    def slot_filling(self, memory):
        if memory["is_repeat"]:
            return memory

        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, memory["query"]):
                    memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory

    def dst(self, memory):
        if memory["is_repeat"]:
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
        if memory["is_repeat"]:
            memory["policy"] = "repeat"
            return memory

        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]
        return memory

    def nlg(self, memory):
        if memory["policy"] == "repeat":
            # 重听功能：返回上一轮的回复
            memory["response"] = memory.get("last_response", "抱歉，没有之前的对话记录")
        elif memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        else:
            slot = memory["require_slot"]
            memory["response"] = self.slot_info[slot]["query"]

        # 保存当前回复，供下一轮重听使用
        memory["last_response"] = memory["response"]
        return memory

    def fill_in_slot(self, template, memory):
        node = memory["hit_node"]
        node_info = self.all_node_info[node]
        for slot in node_info.get("slot", []):
            if slot in memory:
                template = template.replace(slot, memory[slot])
        return template

    def run(self, query, memory):
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    memory = {
        "available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"],
        "last_response": ""
    }

    while True:
        query = input("请输入：")
        memory = ds.run(query, memory)
        print(f"系统回复: {memory['response']}")
        print("=" * 50)
