import json
import pandas as pd
import re


class DialogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.nodes_info = {}
        self.load_scenario("scenario/scenario-买衣服.json")
        self.load_slot_template("scenario/slot_fitting_templet.xlsx")

    def load_scenario(self, scenario_path):
        # 加载场景信息
        with open(scenario_path, 'r', encoding='utf-8') as f:
            self.scenario = json.load(f)
        scenario_name = scenario_path.split('/')[1].split('.')[0]
        for node in self.scenario:
            self.nodes_info[scenario_name + node['id']] = node
            if "childnode" in node:
                node["childnode"] = [scenario_name + childnode for childnode in node["childnode"]]

    def load_slot_template(self, slot_template_path):
        self.slot_template = pd.read_excel(slot_template_path)
        # slot query values
        self.slot_to_qv = {}
        for i, row in self.slot_template.iterrows():
            slot = row["slot"]
            query = row["query"]
            values = row["values"]
            self.slot_to_qv[slot] = [query, values]

    def generate_response(self, query, memory):
        memory['query'] = query
        memory = self.nul(memory)  # 自然语言理解
        memory = self.dst(memory)  # 对话状态识别
        memory = self.dpo(memory)  # 对话策略优化
        memory = self.nlg(memory)  # 自然语言生成
        return memory

    def nul(self, memory):
        memory = self.intent_recognition(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_recognition(self, memory):
        # 意图识别模块 跟available_nodes中每个节点打分，选择分数最高的作为当前节点
        max_score = -1
        for node_name in memory["available_nodes"]:
            node_info = self.nodes_info[node_name]
            score = self.get_node_score(memory["query"], node_info)
            if score > max_score:
                max_score = score
                if node_name == "scenario-买衣服node5":
                    memory["policy"] = "repeat"
                    memory["recent_hit_node"] = memory["hit_node"]
                memory["hit_node"] = node_name
        return memory

    def get_node_score(self, query, node_info):
        # 跟node中的intent算分
        intent_list = node_info["intent"]
        score = 0
        for intent in intent_list:
            score = max(score, self.get_intent_score(query, intent))
        return score

    def get_intent_score(self, string1, string2):
        # 计算两个句子之间的相似度 使用 jaccard距离
        s1 = set(string1)
        s2 = set(string2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def slot_filling(self, memory):
        # 槽位填充模块，根据当前节点中的slot，对query进行槽位填充
        # 根据命中的节点，获取对应的slot
        # ['#服装类型#', '#服装颜色#', '#服装尺寸#']
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        # 对query进行槽位填充
        for slot in slot_list:
            # ['您想买长袖、短袖还是半截袖', '长袖|短袖|半截袖'] -> '长袖|短袖|半截袖'
            slot_values = self.slot_to_qv[slot][1]
            if re.search(slot_values, memory["query"]):
                memory[slot] = re.search(slot_values, memory["query"]).group()
        return memory

    def dst(self, memory):
        # 确认当前hit_node所需要的所有槽位是否已经齐全
        if memory["policy"] == "repeat":
            memory["recent_require_slot"] = memory["require_slot"]
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        # 如果require_slot为空，则执行当前节点的操作,否则进行反问
        if memory["policy"] == "repeat":
            pass
        else:
            if memory["require_slot"] is None:
                memory["policy"] = "reply"
                childnodes = self.nodes_info[memory["hit_node"]].get("childnode", [])
                memory["available_nodes"] = childnodes
            else:
                memory["policy"] = "ask"
                memory["available_nodes"] = [memory["hit_node"], "scenario-买衣服node5"]  # 停留在当前节点，直到槽位填满
        return memory

    def nlg(self, memory):
        # 根据policy生成回复,反问或回复
        if memory["policy"] == "reply":
            response = self.nodes_info[memory["hit_node"]]["response"]
            response = self.fill_in_template(response, memory)
            memory["response"] = response
        elif memory["policy"] == "repeat":
            # 您想要多尺寸
            repeat_response = self.nodes_info[memory["hit_node"]]["response"]
            if repeat_response in memory["response"]:
                memory["response"] = memory["response"]
            else:
                response = repeat_response + "，" + memory["response"]
                memory["response"] = response
            memory["hit_node"] = memory["recent_hit_node"]
            memory["require_slot"] = memory["recent_require_slot"]
            memory["policy"] = "ask"
        else:
            slot = memory["require_slot"]
            # ['您想买长袖、短袖还是半截袖', '长袖|短袖|半截袖'] -> 您想买长袖、短袖还是半截袖
            memory["response"] = self.slot_to_qv[slot][0]
        return memory

    def fill_in_template(self, response, memory):
        slot_list = self.nodes_info[memory["hit_node"]].get("slot", [])
        for slot in slot_list:
            if slot in response:
                response = response.replace(slot, memory[slot])
        return response


if __name__ == '__main__':
    ds = DialogueSystem()
    # {'scenario-买衣服node1': {'id': 'node1', 'intent': ['我要买衣服'],
    #                           'slot': ['#服装类型#', '#服装颜色#', '#服装尺寸#'],
    #                           'action': ['select 衣服 where 类型=#服装类型# and 颜色=#服装颜色# and 尺寸=#服装尺寸#'],
    #                           'response': '为您推荐这一款，#服装尺寸#号，#服装颜色#色#服装类型#，产品连接：xxx',
    #                           'childnode': ['scenario-买衣服node2', 'scenario-买衣服node3', 'scenario-买衣服node4']},
    #  'scenario-买衣服node2': {'id': 'node2', 'intent': ['我没钱'], 'response': '没钱你可以选择分期付款',
    #                           'childnode': ['scenario-买衣服node3']},
    #  'scenario-买衣服node3': {'id': 'node3', 'intent': ['可以分期付款吗'], 'slot': ['#分期付款期数#', '#支付方式#'],
    #                           'action': ['MAKE_PAYMENT'],
    #                           'response': '好的，为您办理分期付款，分#分期付款期数#期，使用#支付方式#支付，谢谢惠顾'},
    #  'scenario-买衣服node4': {'id': 'node4', 'intent': ['我买了'], 'action': ['TAKE_ORDER'],
    #                           'response': '已为您下单，谢谢惠顾，流程结束'}}
    print(ds.nodes_info)
    # {'#服装类型#': ['您想买长袖、短袖还是半截袖', '长袖|短袖|半截袖'],
    #  '#服装颜色#': ['您喜欢什么颜色', '红|橙|黄|绿|青|蓝|紫'], '#服装尺寸#': ['您想要多尺寸', 's|m|l|xl|xll'],
    #  '#分期付款期数#': ['您想分多少期，可以有3期，6期，9期，12期', '3|6|9|12'],
    #  '#支付方式#': ['您想使用什么支付方式', '信用卡|支付宝|微信']}
    #
    print(ds.slot_to_qv)
    memory = {"available_nodes": ["scenario-买衣服node1", "scenario-买衣服node5"], "policy": ""}  # 默认初始记忆为空
    while True:
        print(memory)
        query = input("User: ")
        memory = ds.generate_response(query, memory)
        print("Bot: ", memory["response"])
