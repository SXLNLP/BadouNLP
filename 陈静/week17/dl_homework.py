'''
在原版基础上添加重听功能

新增功能：
1. 智能重听：支持多种重听触发词
2. 历史记录：保存对话历史和回复
3. 智能处理：长内容概括、重复次数限制
4. 用户友好：清晰的重听提示

'''

import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
        # 重听功能相关配置
        self.repeat_keywords = [
            "再说一遍", "重复", "没听清楚", "没听懂", 
            "请再说一次", "能重复吗", "刚才说什么",
            "再说一次", "没听清", "请重复一下", "重听",
            "重复一遍", "请再说", "再说"
        ]
        self.max_repeat_count = 3  # 最大重听次数限制
    
    def load(self):
        self.all_node_info = {}
        self.load_scenario(r"D:\code\ai\week17 对话系统\scenario\scenario-买衣服.json")
        self.load_slot_templet(r"D:\code\ai\week17 对话系统\scenario\slot_fitting_templet.xlsx")

    def load_scenario(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
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
      
    def is_repeat_request(self, query):
        '''检查用户是否要求重听'''
        return any(keyword in query for keyword in self.repeat_keywords)
    
    def handle_repeat_request(self, memory):
        '''处理重听请求'''
        # 检查是否有历史回复
        if "last_response" not in memory or not memory["last_response"]:
            memory["response"] = "抱歉，我没有上次的回复记录。"
            return memory
        
        # 检查重听次数
        if memory.get("repeat_count", 0) >= self.max_repeat_count:
            memory["response"] = f"我已经重复了{self.max_repeat_count}次了，如果您还是没听清楚，建议您换个方式询问。"
            return memory
        
        # 增加重听次数
        memory["repeat_count"] = memory.get("repeat_count", 0) + 1
        
        # 获取上次回复
        last_response = memory["last_response"]
        
        # 智能重听：根据内容长度决定处理方式
        if len(last_response) > 100:
            # 长内容进行简单概括
            summary = self.summarize_response(last_response)
            memory["response"] = f"内容较多，我为您概括一下：{summary}\n\n如果您需要完整内容，请告诉我。"
        else:
            # 短内容直接重复
            memory["response"] = f"好的，我再重复一遍：{last_response}"
        
        # 重听时不改变对话状态，保持原有节点
        return memory
    
    def summarize_response(self, response):
        '''对长回复进行简单概括'''
        # 简单的概括逻辑：提取前50个字符
        if len(response) > 50:
            return response[:50] + "..."
        return response
    
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
                if re.search(slot_values, memory['query']):
                    memory[slot] = re.search(slot_values, memory['query']).group()
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
        
        # 检查是否是重听请求
        if self.is_repeat_request(query):
            return self.handle_repeat_request(memory)
        
        # 正常对话流程
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        
        # 更新历史记录
        memory["last_response"] = memory["response"]
        memory["repeat_count"] = 0  # 重置重听次数
        
        # 添加到对话历史
        if "conversation_history" not in memory:
            memory["conversation_history"] = []
        memory["conversation_history"].append({
            "user_input": query,
            "system_response": memory["response"]
        })
        
        return memory
        

if __name__ == '__main__':
    ds = DialogueSystem()
    
    # 初始化用户状态（增强版）
    memory = {
        "available_nodes": ["scenario-买衣服_node1"],
        "conversation_history": [],  # 对话历史
        "last_response": "",         # 上次回复
        "repeat_count": 0            # 重听次数
    }
    
    print("=== 智能对话系统启动 ===")
    print("支持重听功能，您可以说：再说一遍、重复、没听清楚等")
    print("输入 'quit' 退出对话")
    print("========================")
    
    while True:
        query = input("请输入：")
        
        if query.lower() in ['quit', 'exit', '退出']:
            print("感谢使用，再见！")
            break
        
        if not query.strip():
            print("请输入有效内容")
            continue
        
        memory = ds.run(query, memory)
        
        print(f"\n系统回复：{memory['response']}")
        print(f"重听次数：{memory.get('repeat_count', 0)}")
        print("=" * 50)
        
        # 显示当前对话状态（调试用）
        if memory.get("hit_node"):
            print(f"当前节点：{memory['hit_node']}")
        if memory.get("require_slot"):
            print(f"需要槽位：{memory['require_slot']}")
        print()