import re
import pandas as pd
import json
import os

class main():
    def __init__(self):
        # 所有节点文件地址
        self.node_info_path = ['scenario-买衣服.json', 'scenario-看电影.json']
        # 所有问题模版文件地址
        self.question_template_path = ['slot_fitting_templet.xlsx']

        # 总流程状态管理
        self.status = {
            'available_nodes': [], # 当前可用节点名称
            'available_nodes_init': [], # 初始化节点根目录(用于突然换主体节点问题)

            # 'process': '', # 流程状态
            #     # slot_is_empty 存在待填槽
            #     # slot_is_done 槽已填完
            #     # end 全流程结束
            # 'require_slot': [], # 是否存在待填槽
            # 'question': '', # 客户当前问题
            # 'question_old': '', # 客户主体问题
            # 'hit_root_directory': '', # 当前命中的根目录
            # 'hit_node_info_item': '', # 当前命中的节点名称'
            # 'hit_node_info_item_old': '', # 上一个节点
            # 'slots': {},
        }

        self.node_info_data = {} # 所有节点信息
        self.slot_info_data = {} # 所以问题模版

        self.status_init()
        self.load_data()
        self.process()

    # 初始化部分数据
    def status_init(self):
        self.status['process'] = '' # 流程状态
        self.status['require_slot'] = [] # 是否存在待填槽
        self.status['question'] = '' # 客户当前问题
        self.status['question_old'] = '' # 客户主体问题
        self.status['hit_root_directory'] = '' # 当前命中的根目录
        self.status['hit_node_info_item'] = '' # 当前命中的节点名称
        self.status['hit_node_info_item_old'] = '' # 上一个节点
        self.status['slots'] = {} 

    # 加载节点数据
    def load_data(self):
        print("======================== 开始加载数据 ========================")
        # 加载所有节点
        for i,item in enumerate(self.node_info_path):
            with open(item,'r',encoding='utf-8') as f:
                name = os.path.basename(item).split('.')[0]
                for node in json.load(f):
                    # 可以通过 是否有引用，是否有子节点 来判断是否属于根节点，但目前就用id为1吧
                    if node['id'] == 'node1':
                        self.status['available_nodes'].append(name + '_' + node['id'])
                        self.status['available_nodes_init'].append(name + '_' + node['id'])
                    self.node_info_data[name + '_' + node['id']] = node
                    if 'childnode' in node:
                        childnode = [name + '_' + x for x in node['childnode']]
                        self.node_info_data[name + '_' + node['id']]['childnode'] = childnode
        # 加载所有问题模版
        for i,item in enumerate(self.question_template_path):
            data = pd.read_excel(item)
            for i, row in data.iterrows():
                # 算辽算辽不做隔离辽～
                self.slot_info_data[row['slot']] = {
                    'query': row['query'],
                    'values': row['values']
                }
        # print("所有节点数据:\n", self.node_info_data)
        # print("所有模版数据:\n", self.slot_info_data)
        print("======================== 数据加载完成 ========================")
    # 自然语言理解
    def nlu(self):
        self.domain_identification()
        self.intent_recognition()
        # print(f"意图识别结果-{max_score}:", hit_node_info_item)
        self.slot_extraction()

    # 自然语言理解 - 领域识别
    def domain_identification(self):
        quit = self.status['question']
        nodes = self.status['available_nodes_init']
        max_score = 0.6
        hit_node_info_item= ''
        for _,item in enumerate(nodes):
            node_info = self.node_info_data[item]
            score = self.hit_node_info(quit,node_info['intent'])
            if score > max_score:
                max_score = score
                hit_node_info_item = item
        
        if max_score > 0.6 and self.status['hit_root_directory'] != hit_node_info_item:
            self.status['hit_root_directory'] = hit_node_info_item
            self.status_init()
            self.status['question'] = quit

    # 自然语言理解 - 意图识别
    def intent_recognition(self):
        max_score = 0
        hit_node_info_item= ''
        nodes = self.status['available_nodes']
        query = self.status['question']
        if self.status['question_old']:
            query = self.status['question_old']
        # print(f"\n当前主体问题： {query}\n")

        for i,item in enumerate(nodes):
            node_info = self.node_info_data[item]
            score = self.hit_node_info(query, node_info['intent'])
            if score > max_score:
                max_score = score
                hit_node_info_item = item
        if max_score == 0:
            print("没有找到匹配的节点")
            self.status['process'] = 'error'
        else:
            if self.status['process'] == 'error':
                self.status['process'] = 'attention'
            self.status['hit_node_info_item'] = hit_node_info_item
    # 自然语言理解 - 意图识别 - 节点意图打分
    def hit_node_info(self, query, intents):
        max_score = 0
        for i,intent in enumerate(intents):
            score = self.jaccard_distance(query, intent)
            if score > max_score:
                max_score = score
        return max_score
    # 自然语言理解 - 意图识别 - 节点意图打分 - 使用jaccard距离计算
    def jaccard_distance(self, query, intent):
        intersection = len(set(query).intersection(set(intent)))
        union = len(set(query).union(set(intent)))
        score = intersection / union
        if score >= 0.5:
            return score
        else:
            return 0

    # 自然语言理解 - 槽位提取
    def slot_extraction(self):
        if self.status['process'] == 'error':
            return
        query = self.status['question']
        node_info = self.node_info_data[self.status['hit_node_info_item']]
        for slot in node_info.get('slot', []):
            if slot not in self.status['slots']:
                slot_values = self.slot_info_data[slot]['values']
                if re.search(slot_values, query):
                    self.status['slots'][slot] = re.search(slot_values, query).group()
    # 状态追踪
    def dsp(self):
        if self.status['process'] == 'error':
            return
        node_info = self.node_info_data[self.status['hit_node_info_item']]
        self.status['require_slot'] = []
        for slot in node_info.get('slot', []):
            if slot not in self.status['slots']:
                self.status['process'] = 'slot_is_empty'
                self.status['require_slot'].append(slot)
        if len(self.status['require_slot']) <= 0:
            self.status['process'] = 'slot_is_done'
        # print('追踪未填入的slot', self.status['require_slot'])
    # 策略优化
    def dpo(self):
        if self.status['process'] == 'error':
            return
        # 如果这一环节已完成，则获取下一环节内容，并且去掉当前节点（从上个节点获取所有子节点去除）
        if self.status['process'] == 'slot_is_done':
            self.dep_slot_is_done()
        else:
            self.dep_slot_is_empty()
        # print('处理完了策略优化的question_old：\n', self.status['question_old'])
        # print('处理完了策略优化的节点状态：\n', self.status['available_nodes'])
        # print("======================== 历史分析结果 ========================")
        # print('处理完了策略优化的slots：\n', self.status['slots'])
    
    # 策略优化 - 槽位已填入完成
    def dep_slot_is_done(self):
        # print('进了- 1 -节点：完成了父节点 - 没有插槽状态')
        # 判断是否有父节点
        if self.status['hit_node_info_item_old']:
            # print('- 1 - 节点延伸 - 删除父节点逻辑')
            childnode = self.node_info_data[self.status['hit_node_info_item_old']].get('childnode', [])
            [self.status['available_nodes'].remove(x) for x in childnode]
            self.status['hit_node_info_item_old'] = ''
        else:
            # print('- 1 - 节点延伸 - 删除当前节点连接')
            self.status['available_nodes'].remove(self.status['hit_node_info_item'])
        # 将当前命中节点记录下来
        self.status['hit_node_info_item_old'] = self.status['hit_node_info_item']
        # 获取子节点
        childnode = self.node_info_data[self.status['hit_node_info_item']].get('childnode', [])
        # 将子节点加入可用节点列表
        [self.status['available_nodes'].append(x) for x in childnode]
        # 清空命中节点记录
        self.status['hit_node_info_item'] = ''
        # 清空旧问题主体
        self.status['question_old'] = ''
        if len(childnode) <= 0:
            self.status['process'] = 'end'
    # 策略优化 - 槽位未填入完成
    def dep_slot_is_empty(self):
        # print('进了- 2 -节点：没完成当前节点 - 还要插槽状态')
        # 判断是否有父节点 - 清楚父节点下的自节点并且清空父节点记录
        if self.status['hit_node_info_item_old']:
            # print('- 2 - 节点延伸 - 删除父节点逻辑')
            childnode = self.node_info_data[self.status['hit_node_info_item_old']].get('childnode', [])
            [self.status['available_nodes'].remove(x) for x in childnode]
            self.status['hit_node_info_item_old'] = ''
        # 当前命中节点
        hit = self.status['hit_node_info_item']
        # 当前命中节点不在可用节点列表中时，添加当前命中节点
        if hit not in self.status.get('available_nodes', []):
            self.status['available_nodes'].append(hit)
        # 当主体问题不存在时，添加当前主体问题
        if self.status['question_old'] == '':
            # print('- 2 - 节点延伸 - 修改主体名字')
            self.status['question_old'] = self.status['question']
    # 自然语言生成
    def nlg(self):
        print("======================== 模型回答 ========================")
        if self.status['process'] == 'slot_is_done' or self.status['process'] == 'end' or self.status['hit_node_info_item_old']:
            node_info = self.node_info_data[self.status['hit_node_info_item_old']]
            print(self.fill_in_slot(node_info))
        elif self.status['process'] == 'error':
            print('请重新输入您的需求')
        else:
            slot = self.status['require_slot'][0]
            problem = self.slot_info_data[slot]
            print(problem['query'])
    # 自然语言生成 - 填充槽位回复语句
    def fill_in_slot(self, node_info):
        template = node_info['response']
        for slot in node_info.get("slot", []):
            template = template.replace(slot, self.status['slots'][slot])
        return template


    # 问答主流程loop
    def process(self):
        while True:
            # question = print("请输入问题：")
            self.status['question'] = input("请输入:")
            self.main()
            if self.status['process'] == 'end':
                print("======================== 模型结束 ========================")
                print('最后收集的slots：\n', self.status['slots'])
                break
    # 模型主流程
    def main(self):
        self.nlu()
        self.dsp()
        self.dpo()
        self.nlg()
    

if __name__ == "__main__":
    demo = main()