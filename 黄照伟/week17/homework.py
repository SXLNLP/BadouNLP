'''
任务型多轮对话
读取场景脚本完成多轮对话

'''
import json
import pandas as pd
import re
import os

class DialogueSystem:
    def __init__(self):
        self.load()

    def load(self):
        self.all_node_info ={}
        self.load_scenario("/Library/workerspace/python_test/badou2/week17/scenario-买衣服.json")
        self.load_scenario("/Library/workerspace/python_test/badou2/week17/scenario-看电影.json")
        self.load_slot_template("/Library/workerspace/python_test/badou2/week17/slot_fitting_templet.xlsx")


    def load_scenario(self,file):
        with open(file,'r',encoding='utf-8') as f:
            scenario = json.load(f)
        scenario_name = os.path.basename(file).split('.')[0]
        for node in scenario :
            self.all_node_info[scenario_name + '_' +node['id']] = node
            if 'childnode' in node:
                self.all_node_info[scenario_name + '_' + node['id']]['childnode'] = [scenario_name + '_' + x for x in node['childnode']]
    def load_slot_template(self,file):
        self.slot_template = pd.read_excel(file)
        # 三列 : slot , query , values
        self.slot_info = {}
        #逐行读取,slot 为 key ,query 和 values 为 value
        for i in range(len(self.slot_template)):
            slot = self.slot_template.iloc[i]['slot']
            query = self.slot_template.iloc[i]['query']
            values = self.slot_template.iloc[i]['values']
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values

    def nlu(self,memory):
        memory = self.intent_judge(memory)
        memory = self.slot_filling(memory)
        return memory

    def intent_judge(self,memory):
        #意图识别,匹配当前可以访问的节点
        query = memory['query']
        max_score = -1
        hit_node = None
        for node in memory['available_nodes']:
            score = self.calculate_node_score(query,node)
            if score > max_score:
                max_score = score
                hit_node = node
        memory['hit_node'] = hit_node
        memory['intent_score'] = max_score
        return memory
    def calculate_node_score(self,query,node):
        node_info = self.all_node_info[node]
        intent = node_info['intent']
        max_score = -1
        for sentence in intent:
            score = self.calculate_sentence_score(query,sentence)
            if score > max_score:
                max_score = score
        return max_score

    def calculate_sentence_score(self,query,sentence):
        query_words = set(query)
        sentence_words = set(sentence)
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        return len(intersection) / len(union)

    def slot_filling(self,memory):
        hit_node = memory['hit_node']
        node_info = self.all_node_info[hit_node]
        for slot in node_info.get('slot',[]):
            if slot not in memory:
                slot_values = self.slot_info[slot]['values']
                if re.search(slot_values,query):
                    memory[slot] = re.search(slot_values,query).group()
        return memory

    def dst(self,memory):
        hit_node = memory['hit_node']
        node_info = self.all_node_info[hit_node]
        slot = node_info.get('slot',[])
        for s in slot:
            if s not in memory:
                memory['require_slot'] = s
                return memory
        memory['require_slot'] = None
        return memory

    def dpo(self,memory):
        if memory['require_slot'] is None:
            #没有需要填充的槽位
            memory['policy'] = 'reply'
            #self.take_action(memory)
            hit_node = memory['hit_node']
            node_info = self.all_node_info[hit_node]
            memory['available_nodes'] = node_info.get('childnode',[])
        else:
            #有欠缺的槽位
            memory['policy'] = 'request'
            #停留在当前节点
            memory['available_nodes'] = [memory['hit_node']]
        return memory

    def nlg(self,memory):
        #根据policy执行反问 或 回答
        if memory['policy'] == 'reply':
            hit_node = memory['hit_node']
            node_info = self.all_node_info[hit_node]
            memory['response'] = self.fill_in_slot(node_info['response'],memory)

        else:
            #policy == 'request'
            slot = memory['require_slot']
            memory['response'] = self.slot_info[slot]['query']
        return memory
    def fill_in_slot(self,template,memory):
        node = memory['hit_node']
        node_info = self.all_node_info[node]
        for slot in node_info.get('slot',[]):
            template = template.replace(slot,memory[slot])
        return template

    def run(self,query,memory):
        '''
        :param query: 用户输入
        :param memory: 用户状态
        :return:
        '''
        memory['query'] = query
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.dpo(memory)
        memory = self.nlg(memory)
        return memory

if __name__ == '__main__':
    ds = DialogueSystem()
    print(ds.slot_info)

    memory = {'available_nodes':['scenario-买衣服_node1','scenario-看电影_node1']}
    while True:
        query = input('请输入:')

        memory = ds.run(query,memory)
        print(memory)
        print()
        response = memory['response']
        print(response)
        print("============================")
