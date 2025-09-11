import json
import pandas as pd
import re
import os
from typing import Dict, List, Any, Optional


class DialogueSystem:
    def __init__(self):
        self.all_node_info: Dict[str, Dict] = {}
        self.slot_templet: Optional[pd.DataFrame] = None
        self.slot_info: Dict[str, Dict] = {}
        self.load()

    def load(self):
        """加载所有场景和槽位模板"""
        self.load_scenario("scenario-买衣服.json")
        self.load_scenario("scenario-看电影.json")
        self.load_slot_templet("slot_fitting_templet.xlsx")

    def load_scenario(self, file_path: str):
        """加载场景文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                scenario = json.load(f)
            
            scenario_name = os.path.basename(file_path).split('.')[0]
            
            for node in scenario:
                node_id = f"{scenario_name}_{node['id']}"
                self.all_node_info[node_id] = node.copy()  # 使用副本避免修改原始数据
                
                # 处理子节点ID
                if "childnode" in node:
                    self.all_node_info[node_id]['childnode'] = [
                        f"{scenario_name}_{child_id}" for child_id in node['childnode']
                    ]
                    
        except FileNotFoundError:
            print(f"警告：场景文件 {file_path} 未找到")
        except json.JSONDecodeError:
            print(f"错误：场景文件 {file_path} JSON格式错误")

    def load_slot_templet(self, file_path: str):
        """加载槽位模板"""
        try:
            self.slot_templet = pd.read_excel(file_path)
            self.slot_info = {}
            
            for _, row in self.slot_templet.iterrows():
                slot = row['slot']
                query = row['query']
                values = row['values']
                
                if slot not in self.slot_info:
                    self.slot_info[slot] = {}
                
                self.slot_info[slot]['query'] = query
                self.slot_info[slot]['values'] = values
                
        except FileNotFoundError:
            print(f"警告：槽位模板文件 {file_path} 未找到")
        except Exception as e:
            print(f"加载槽位模板时出错: {e}")

    def is_repeat_request(self, query: str) -> bool:
        """判断是否为重听请求"""
        repeat_patterns = [
            "重复", "再说一遍", "重听", "没听清", 
            "没听懂", "刚才说什么", "上一句"
        ]
        query = query.strip().lower()
        
        for pattern in repeat_patterns:
            if pattern in query:
                return True
        return False

    def nlu(self, memory: Dict) -> Dict:
        """自然语言理解"""
        memory = self.intent_judge(memory)
        if not memory["is_repeat"]:
            memory = self.slot_filling(memory)
        return memory

    def intent_judge(self, memory: Dict) -> Dict:
        """意图识别"""
        query = memory['query']

        # 检查重听请求
        if self.is_repeat_request(query):
            memory["is_repeat"] = True
            return memory

        memory["is_repeat"] = False
        max_score = -1
        hit_node = None
        
        for node_id in memory["available_nodes"]:
            if node_id not in self.all_node_info:
                continue
                
            score = self.calculate_node_score(query, node_id)
            if score > max_score:
                max_score = score
                hit_node = node_id

        memory["hit_node"] = hit_node
        memory["intent_score"] = max_score
        return memory

    def calculate_node_score(self, query: str, node_id: str) -> float:
        """计算节点匹配分数"""
        node_info = self.all_node_info.get(node_id, {})
        intent = node_info.get('intent', [])
        
        if not intent:
            return 0.0
            
        max_score = 0.0
        for sentence in intent:
            score = self.calculate_sentence_similarity(query, sentence)
            if score > max_score:
                max_score = score
                
        return max_score

    def calculate_sentence_similarity(self, query: str, sentence: str) -> float:
        """计算句子相似度"""
        if not query or not sentence:
            return 0.0
            
        query_words = set(query)
        sentence_words = set(sentence)
        
        if not query_words or not sentence_words:
            return 0.0
            
        intersection = query_words.intersection(sentence_words)
        union = query_words.union(sentence_words)
        
        return len(intersection) / len(union) if union else 0.0

    def slot_filling(self, memory: Dict) -> Dict:
        """槽位填充"""
        hit_node = memory.get("hit_node")
        if not hit_node or hit_node not in self.all_node_info:
            return memory
            
        node_info = self.all_node_info[hit_node]
        slots = node_info.get('slot', [])
        
        for slot in slots:
            if slot not in memory and slot in self.slot_info:
                slot_values = self.slot_info[slot].get("values", "")
                if slot_values and re.search(slot_values, memory["query"]):
                    match = re.search(slot_values, memory["query"])
                    if match:
                        memory[slot] = match.group()
                        
        return memory

    def dst(self, memory: Dict) -> Dict:
        """对话状态跟踪"""
        if memory["is_repeat"]:
            return memory

        hit_node = memory.get("hit_node")
        if not hit_node or hit_node not in self.all_node_info:
            memory["require_slot"] = None
            return memory
            
        node_info = self.all_node_info[hit_node]
        slots = node_info.get('slot', [])
        
        for slot in slots:
            if slot not in memory:
                memory["require_slot"] = slot
                return memory
                
        memory["require_slot"] = None
        return memory

    def dpo(self, memory: Dict) -> Dict:
        """对话策略优化"""
        if memory["is_repeat"]:
            memory["policy"] = "repeat"
            return memory

        if memory.get("require_slot") is None:
            memory["policy"] = "reply"
            hit_node = memory.get("hit_node")
            if hit_node and hit_node in self.all_node_info:
                node_info = self.all_node_info[hit_node]
                memory["available_nodes"] = node_info.get("childnode", [])
            else:
                memory["available_nodes"] = []
        else:
            memory["policy"] = "request"
            memory["available_nodes"] = [memory.get("hit_node", "")]
            
        return memory

    def nlg(self, memory: Dict) -> Dict:
        """自然语言生成"""
        if memory["policy"] == "repeat":
            # 重听功能：返回上一轮的回复
            memory["response"] = memory.get("last_response", "抱歉，没有之前的对话记录")
            
        elif memory["policy"] == "reply":
            hit_node = memory.get("hit_node")
            if hit_node and hit_node in self.all_node_info:
                node_info = self.all_node_info[hit_node]
                memory["response"] = self.fill_in_slot(node_info.get("response", ""), memory)
            else:
                memory["response"] = "抱歉，我没有理解您的意思"
                
        else:  # request policy
            slot = memory.get("require_slot")
            if slot and slot in self.slot_info:
                memory["response"] = self.slot_info[slot].get("query", f"请提供{slot}信息")
            else:
                memory["response"] = "请提供相关信息"

        # 保存当前回复，供下一轮重听使用
        memory["last_response"] = memory["response"]
        return memory

    def fill_in_slot(self, template: str, memory: Dict) -> str:
        """填充槽位到模板"""
        if not template:
            return ""
            
        hit_node = memory.get("hit_node")
        if not hit_node or hit_node not in self.all_node_info:
            return template
            
        node_info = self.all_node_info[hit_node]
        slots = node_info.get("slot", [])
        
        for slot in slots:
            if slot in memory:
                template = template.replace(slot, str(memory[slot]))
                
        return template

    def run(self, query: str, memory: Dict) -> Dict:
        """运行对话系统"""
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
        "last_response": "",
        "is_repeat": False
    }

    print("对话系统已启动，输入'退出'结束对话")
    print("=" * 50)
    
    while True:
        try:
            query = input("用户输入: ").strip()
            if query == "退出":
                print("对话结束")
                break
                
            if not query:
                continue
                
            memory = ds.run(query, memory)
            print(f"系统回复: {memory['response']}")
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n对话结束")
            break
        except Exception as e:
            print(f"系统错误: {e}")
            print("=" * 50)
