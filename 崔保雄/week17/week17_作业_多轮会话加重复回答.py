

'''
任务型多轮对话系统  -- 作业
改动代码的地方：
1、__init__函数追加 self.repeat_intent = ["请再说一遍", "再重复一遍", "重复一遍", "再来一遍"]，列举了识别用户希望重复回答的可能句子
    也是使用jaccard距离计算相似度
2、intent_judge(self, memory)方法里追加以下代码
            ###########################################  这里的代码用来判断是否需要重复回复  Start #############################
            #计算用户查询文本，跟全局重复回答意图句子
            max_score_repeat = -100
            #self.calucate_sentence_score(query, self.repeat_intent)
            for repeat_sentence in self.repeat_intent:
                score_repeat = self.calucate_sentence_score(query, repeat_sentence)
                if score_repeat > max_score_repeat:
                    max_score_repeat = score_repeat
            if max_score_repeat > score:
                print("#################需要重复回答#################")
                return memory
            ###########################################  这里的代码用来判断是否需要重复回复  End ##############################
'''

import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()
        #识别用户希望重复回答的可能句子
        self.repeat_intent = ["请再说一遍", "再重复一遍", "重复一遍", "再来一遍"]
    
    def load(self):
        self.all_node_info = {}
        self.load_scenario("scenario-买衣服.json")
        # self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")


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



    #计算查询词跟指定节点意图的相似度分数
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

    #使用jaccard距离计算两个句子的相似度
    def calucate_sentence_score(self, query, sentence):
        #两个字符串做文本相似度计算。jaccard距离计算相似度
        query_words = set(query)
        sentence_words = set(sentence)
        #两个句子字符交集
        intersection = query_words.intersection(sentence_words)
        #两个句子字符并集
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)


    #意图识别，从当前可用节点中，找到一个最匹配的节点
    def intent_judge(self, memory):
        #意图识别，匹配当前可以访问的节点
        query = memory['query']
        max_score = -1
        hit_node = None #命中节点
        for node in memory["available_nodes"]:
            score = self.calucate_node_score(query, node)

            ###########################################  这里的代码用来判断是否需要重复回复  Start #############################
            #计算用户查询文本，跟全局重复回答意图句子
            max_score_repeat = -100
            #self.calucate_sentence_score(query, self.repeat_intent)
            for repeat_sentence in self.repeat_intent:
                score_repeat = self.calucate_sentence_score(query, repeat_sentence)
                if score_repeat > max_score_repeat:
                    max_score_repeat = score_repeat
            if max_score_repeat > score:
                print("#################需要重复回答#################")
                return memory
            ###########################################  这里的代码用来判断是否需要重复回复  End ##############################

            if score > max_score:
                max_score = score
                hit_node = node
        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    #槽位填充，将查询词，跟当前匹配的节点的slot的值进行匹配
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

    #NLU（意图识别，识别用户要干啥）
    def nlu(self, memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    #DST（语义槽填充-状态追踪）
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

    #DPO（语义槽填充-策略优化）
    def dpo(self, memory):
        if memory["require_slot"] is None:
            #没有需要填充的槽位，直接回答
            memory["policy"] = "reply"
            # self.take_action(memory)
            hit_node = memory["hit_node"]
            node_info = self.all_node_info[hit_node]
            memory["available_nodes"] = node_info.get("childnode", [])
        else:
            #有欠缺的槽位，需要反问
            memory["policy"] = "request"
            memory["available_nodes"] = [memory["hit_node"]] #停留在当前节点
        return memory

    #NLG（自然语言生成） 根据policy执行反问或回答
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

        # NLU（意图识别，识别用户要干啥）
        memory = self.nlu(memory)

        # DST（语义槽填充-状态追踪）
        memory = self.dst(memory)

        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory
        

if __name__ == '__main__':
    ds = DialogueSystem()
    # print(ds.all_node_info)
    print("ds.slot_info：\n", ds.slot_info)

    # memory = {"available_nodes":["scenario-买衣服_node1","scenario-看电影_node1"]} #用户状态
    memory = {"available_nodes": ["scenario-买衣服_node1"]}  # 用户状态
    while True:
        query = input("请输入：")
        # query = "你好"    
        memory = ds.run(query, memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("===========")
        





