"""
对话系统
描述：读取场景脚本完成多轮对话
"""
import json
import os.path
import re

import pandas as pd


class DialogSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.all_node_info = {}
        self.load_scenario("data/scenario-买衣服.json")
        self.load_scenario("data/scenario-看电影.json")
        self.load_slot_templet("data/slot_fitting_templet.xlsx")

    def load_slot_templet(self, slot_templet_path):
        self.slot_templet = pd.read_excel(slot_templet_path)
        # 三列：slot、query、values
        self.slot_info = {}
        # 遍历slot_templet
        for index, row in self.slot_templet.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_info[slot] = {"query": query, "values": values}

    def load_scenario(self, scenario_path):
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario = json.load(f)
        senario_name = os.path.basename(scenario_path).split(".")[0]
        for node_info in scenario:
            self.all_node_info[senario_name + "_" + node_info["id"]] = node_info

            if "childnode" in node_info:
                self.all_node_info[senario_name + "_" + node_info["id"]]["childnode"] = \
                    [senario_name + "_" + x for x in node_info["childnode"]]

    def run(self, query, memory):
        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.lng(memory)
        return memory

    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory):
        # 意图识别，匹配当前可以访问的节点
        # 判断用户输入是否重听
        if self.repeat(memory):
            return memory
        max_score = -1
        hit_node = None
        for node in memory["available_nodes"]:
            score = self.calc_node_score(query, node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    def repeat(self, memory):
        with open("data/重听.json", "r", encoding="utf-8") as f:
            repeat_info = json.load(f)
        intents = repeat_info["intent"]
        min_score = repeat_info["min_score"]
        query = memory["query"]
        for intent in intents:
            score = self.jaccard_similarity(query, intent)
            if score > min_score:
                return True
        return False

    def calc_node_score(self, query, node):
        # 节点意图打分，计算query和intent的相似度
        intents = self.all_node_info[node]["intent"]
        max_score = -1
        for intent in intents:
            score = self.jaccard_similarity(query, intent)
            if score > max_score:
                max_score = score
        return max_score

    def jaccard_similarity(self, str1, str2):
        # 计算两个字符串的jaccard相似度
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def slot_filling(self, memory):
        # 槽位填充
        query = memory["query"]
        hit_node = memory["hit_node"]
        print(hit_node)
        node_info = self.all_node_info[hit_node]
        # 命中节点有槽位，且槽位未填充
        for slot in node_info.get("slot", []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, query):
                    memory[slot] = re.search(slot_values, query).group()
        return memory

    def dst(self, memory):
        # 状态跟踪，判断槽位是否都已填充
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        # 遍历slot
        for slot in node_info.get("slot", []):
            if slot not in memory:
                # 槽位未填充
                memory["required_slots"] = slot
                return memory
        # 所有槽位都已填充
        memory["required_slots"] = None
        return memory

    def dpo(self, memory):
        # 计划生成
        required_slots = memory["required_slots"]
        if required_slots is not None:
            # 槽位未填充
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]
        else:
            # 所有槽位都已填充
            memory["policy"] = "confirm"
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        return memory

    def lng(self, memory):
        # 语言生成
        policy = memory["policy"]
        if policy == "ask":
            # 槽位未填充
            slot = memory["required_slots"]
            memory["response"] = self.slot_info[slot]["query"]
        elif policy == "confirm":
            # 所有槽位都已填充
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["response"] = self.fill_in_slot(node_info["response"], memory)
        return memory

    def fill_in_slot(self, response, memory):
        # 填充槽位
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get("slot", []):
            if slot in memory:
                response = response.replace(slot, memory[slot])
        return response


if __name__ == "__main__":
    dialog_system = DialogSystem()
    # print(dialog_system.slot_info)
    print(dialog_system.all_node_info)

    memory = {"available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"]}
    while True:
        query = input("请输入：")
        memory = dialog_system.run(query, memory)
        print(memory)
        response = memory['response']
        print(response)
        print("===================")
        if not memory["available_nodes"]:
            break
