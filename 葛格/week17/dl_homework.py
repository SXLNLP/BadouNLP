

'''
任务型多轮对话系统
读取场景脚本完成多轮对话

多加一个功能：
用户可能在任意节点输入“没听清，请再说一遍”这样的请求
要求系统在接收到这类请求后，重新问一下上次的问题，然后整个流程不间断
'''

import json
import jieba
import pandas as pd
from collections import defaultdict

class DialogueSystem:
    def __init__(self):
        self.all_node_info = {}  # 存储所有场景的节点信息
        self.slot_info = {}     # 存储槽位信息
        self.load()
        # 新增：存储上一次的回复
        self.last_response = None
        
    def load(self):
        # 加载场景脚本和槽位模板（保持不变）
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_templet.xlsx")
        
    # 其他原有方法保持不变：load_scenario, load_slot_templet 等
    
    def nlu(self, query, memory):
        """自然语言理解：识别意图和填充槽位"""
        # 新增：检查是否是"没听清"类意图
        if self.is_repeat_intent(query):
            return {
                "intent": "repeat",
                "slot": memory.get("slot", {}),
                "hit_node": memory.get("current_node")
            }
            
        # 原有意图识别逻辑
        available_nodes = memory.get("available_nodes", [])
        hit_node, score = self.intent_judge(query, available_nodes)
        slot = self.slot_filling(query, hit_node) if hit_node else {}
        
        return {
            "intent": "normal",
            "slot": slot,
            "hit_node": hit_node
        }
    
    def is_repeat_intent(self, query):
        """判断用户是否表达了没听清需要重复的意图"""
        # 关键词列表，可以根据需要扩展
        repeat_keywords = ["没听清", "再说一遍", "你说什么", "重复", "没听懂"]
        query_words = set(jieba.cut(query))
        # 如果包含任何一个关键词，判断为需要重复
        for keyword in repeat_keywords:
            if keyword in query_words:
                return True
        return False
    
    def dpo(self, nlu_result, dst_result, memory):
        """对话策略优化"""
        # 新增：处理重复意图
        if nlu_result["intent"] == "repeat":
            return {
                "action": "repeat",
                "available_nodes": memory.get("available_nodes", [])
            }
        
        # 原有策略逻辑
        if dst_result["require_slot"] is None:
            # 槽位已填满，进入下一个节点
            current_node = nlu_result["hit_node"]
            next_nodes = self.all_node_info.get(current_node, {}).get("children", [])
            return {
                "action": "reply",
                "available_nodes": next_nodes
            }
        else:
            # 需要追问，保持当前节点
            return {
                "action": "request",
                "available_nodes": memory.get("available_nodes", [])
            }
    
    def nlg(self, dpo_result, dst_result, nlu_result, memory):
        """自然语言生成"""
        # 新增：处理重复动作
        if dpo_result["action"] == "repeat":
            # 如果有上一次回复，就返回它，否则返回默认提示
            if self.last_response:
                return self.last_response
            else:
                return "请问有什么可以帮助您的吗？"
        
        # 原有生成逻辑
        if dpo_result["action"] == "reply":
            current_node = nlu_result["hit_node"]
            response_template = self.all_node_info.get(current_node, {}).get("response", "")
            response = self.fill_in_slot(response_template, memory["slot"])
        else:  # request
            slot_name = dst_result["require_slot"][0]
            response = self.slot_info.get(slot_name, {}).get("query", "请问您的" + slot_name + "是什么？")
        
        # 新增：保存当前回复作为下一次可能的重复内容
        self.last_response = response
        return response
    
    # 其他原有方法保持不变：intent_judge, calucate_node_score, 
    # calucate_sentence_score, slot_filling, dst, fill_in_slot, run 等
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
        return len(intersection) / len(union)


    def slot_filling(self, memory):
        #槽位填充
        hit_node = memory["hit_node"]
        node_info = self.all_node_info[hit_node]
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
    memory = {
        "available_nodes": ["买衣服_0", "看电影_0"],
        "slot": {},
        "current_node": None
    }
    
    while True:
        query = input("请输入：")    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
