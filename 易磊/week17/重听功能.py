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
        # 添加重听意图关键词
        self.replay_keywords = {"重听", "再说一遍", "重复", "再说一次"}

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
        # 先检查是否是重听请求
        if self.check_replay_request(memory['query']):
            memory['is_replay'] = True
            return memory

        memory['is_replay'] = False
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def check_replay_request(self, query):
        """检查用户输入是否为重听请求"""
        query_words = set(query)
        # 如果用户输入包含任何重听关键词，则判断为重听请求
        return len(query_words & self.replay_keywords) > 0

    def intent_judge(self, memory):
        # 意图识别，匹配当前可以访问的节点
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
        return len(intersection) / len(union) if len(union) > 0 else 0

    def slot_filling(self, memory):
        # 槽位填充
        hit_node = memory["hit_node"]
        if not hit_node:
            return memory

        node_info = self.all_node_info[hit_node]
        query = memory['query']
        for slot in node_info.get('slot', []):
            if slot not in memory:
                slot_values = self.slot_info[slot]["values"]
                if re.search(slot_values, query):
                    memory[slot] = re.search(slot_values, query).group()
        return memory

    def dst(self, memory):
        # 如果是重听请求，不需要处理对话状态
        if memory.get('is_replay', False):
            return memory

        hit_node = memory["hit_node"]
        if not hit_node:
            memory["require_slot"] = None
            return memory

        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot', [])
        for s in slot:
            if s not in memory:
                memory["require_slot"] = s
                return memory
        memory["require_slot"] = None
        return memory

    def dpo(self, memory):
        # 如果是重听请求，保持当前状态不变
        if memory.get('is_replay', False):
            memory["policy"] = "replay"
            return memory

        if memory["require_slot"] is None:
            # 没有需要填充的槽位
            memory["policy"] = "reply"
            hit_node = memory["hit_node"]
            if hit_node and hit_node in self.all_node_info:
                node_info = self.all_node_info[hit_node]
                memory["available_nodes"] = node_info.get("childnode", [])
        else:
            # 有欠缺的槽位
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]]  # 停留在当前节点
        return memory

    def nlg(self, memory):
        # 如果是重听请求，返回上一次的响应
        if memory.get('is_replay', False):
            if 'history_responses' in memory and len(memory['history_responses']) > 0:
                memory["response"] = memory['history_responses'][-1]
            else:
                memory["response"] = "之前没有对话记录可以重听哦~"
            return memory

        # 根据policy执行反问或回答
        if memory["policy"] == "reply":
            hit_node = memory["hit_node"]
            if hit_node and hit_node in self.all_node_info:
                node_info = self.all_node_info[hit_node]
                memory["response"] = self.fill_in_slot(node_info["response"], memory)
            else:
                memory["response"] = "抱歉，我不太明白你的意思"
        else:
            # policy == "request"
            slot = memory["require_slot"]
            if slot and slot in self.slot_info:
                memory["response"] = self.slot_info[slot]["query"]
            else:
                memory["response"] = "抱歉，我需要更多信息"
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
        # 确保memory中有历史记录字段
        if 'history_responses' not in memory:
            memory['history_responses'] = []

        memory["query"] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)

        # 如果不是重听请求，将当前响应添加到历史记录
        if not memory.get('is_replay', False):
            memory['history_responses'].append(memory["response"])

        return memory


if __name__ == '__main__':
    ds = DialogueSystem()
    # print(ds.all_node_info)
    print(ds.slot_info)

    # 初始化memory，添加历史记录字段
    memory = {
        "available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"],
        "history_responses": []  # 用于存储历史响应
    }

    while True:
        query = input("请输入：")
        if query.lower() in ["exit", "quit", "退出"]:
            print("对话结束")
            break

        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
