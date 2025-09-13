'''
找一个编程辅助工具
我的：codegeex

编程建议：
1、不要先直接写逻辑，先写测试
2、先写主流程框架，不要深入细节
3、一个方法尽量不超过20行代码

基于脚本的任务型对话系统

'''
import json
import pandas as pd
import re


class DailogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.nodes_info = {}
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_template("slot_fitting_templet.xlsx")

    def load_scenario(self, scenario_file):
        with open(scenario_file, "r", encoding="utf-8") as f:
            self.scenario = json.load(f)
        scenario_name = scenario_file.split(".")[0]
        for node in self.scenario:
            self.nodes_info[scenario_name + node["id"]] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + childnode for childnode in node["childnode"]]

    def load_slot_template(self, slot_template_file):
        self.slot_template = pd.read_excel(slot_template_file)
        # slot	query	values
        self.slot_to_qv = {}
        for i, row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            # if slot not in self.slot_to_qv:
            #     self.slot_to_qv[slot] = []
            self.slot_to_qv[slot] = [query, values]

    def nlu(self, memory):
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        # 意图识别模块，跟available_nodes中每个节点打分，选择分数最高的作为当前节点
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                memory["hit_node"] = node_name
        return memory

    def get_node_score(self, query, node_info):
        # 跟node中的intent算分
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score, self.sentence_match_score(query, intent))
        return score

    def sentence_match_score(self, string1, string2):
        # 计算两个句子之间的相似度,使用jaccard距离
        s1 = set(string1)
        s2 = set(string2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def slot_filling(self, memory):
        # 槽位填充模块，根据当前节点中的slot，对query进行槽位填充
        # 根据命中的节点，获取对应的slot
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        # 对query进行槽位填充
        for slot in slot_list:
            slot_values = self.slot_to_qv[slot][1]
            if re.search(slot_values, memory["query"]):
                memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory

    def dst(self, memory):
        # 确认当前hit_node所需要的所有槽位是否已经齐全
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        # 如果require_slot为空，则执行当前节点的操作,否则进行反问
        if memory["require_slot"] is None:
            memory["policy"] = "reply"
            childnodes = self.nodes_info[memory["hit_node"]].get("childnode", [])
            memory["available_nodes"] = childnodes
            # 执行动作   take action
        else:
            memory["policy"] = "ask"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点，直到槽位填满
        return memory

    def nlg(self, memory):
        # 根据policy生成回复,反问或回复
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response, memory)
            memory["response"] = response
        else:
            slot = memory["require_slot"]
            memory["response"] = self.slot_to_qv[slot][0]
        return memory

    def fill_in_template(self, response, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response

    def check_rewrite_slot(self, memory):
        """
        如果用户在query中表达了修改槽位的意图（如"我不要红色的了，我想要蓝色"），
        则更新对应的槽值。
        """
        query = memory["query"]
        for slot, (ask_query, pattern) in self.slot_to_qv.items():
            match = re.search(pattern, query)
            if match:
                value = match.group()
                if slot in memory and memory[slot] != value:
                    print(f"[重写提示] 槽位 {slot} 从 {memory[slot]} 修改为 {value}")
                    memory[slot] = value
        return memory

    def generate_response(self, query, memory):
        memory["query"] = query
        memory = self.check_rewrite_slot(memory)
        memory = self.nlu(memory)
        # print(memory)s
        memory = self.dst(memory)  # dialogue state tracking
        memory = self.dpo(memory)  # dialogue policy optimization
        memory = self.nlg(memory)  # natural language generation
        return memory


if __name__ == '__main__':
    ds = DailogueSystem()
    print(ds.slot_to_qv)
    memory = {"available_nodes": ["scenario-买衣服node1", "scenario-看电影node1"]}  # 默认初始记忆为空
    while True:
        # query = "你好，我想订一张从北京到上海的机票"
        query = input("User：")
        memory = ds.generate_response(query, memory)  # memory经常成为dialogue state
        print("System:", memory["response"])

