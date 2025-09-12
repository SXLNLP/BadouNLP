"""
基于BERT的带重听功能的多轮对话系统
实现槽位填充功能

"""

import json
import pandas as pd
import re
import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import Dict, List, Optional, Tuple


class BertDialogueSystem:
    
    def __init__(self, model_name: str = "/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese"):

        self.model_name = model_name
        self.tokenizer = None
        self.bert_model = None
        self.all_node_info = {}
        self.slot_info = {}
        self.conversation_history = []
        
        self._load_bert_model()
        self._load_scenarios()
        self._load_slot_templates()
    
    def _load_bert_model(self):
        """加载BERT模型和分词器"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.bert_model.eval()
    
    def _load_scenarios(self):
        """加载场景配置文件"""
        scenario_files = [
            "scenario/scenario-买衣服.json"
        ]
        
        for file_path in scenario_files:
            if os.path.exists(file_path):
                self._load_scenario(file_path)
    
    def _load_scenario(self, file_path: str):
        """加载单个场景文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            scenario = json.load(f)
        
        scenario_name = os.path.basename(file_path).split('.')[0]
        for node in scenario:
            node_id = f"{scenario_name}_{node['id']}"
            self.all_node_info[node_id] = node
            
            if "childnode" in node:
                child_nodes = [f"{scenario_name}_{x}" for x in node['childnode']]
                self.all_node_info[node_id]['childnode'] = child_nodes
    
    def _load_slot_templates(self):
        """加载槽位模板"""
        template_file = "scenario/slot_fitting_templet.xlsx"
        if os.path.exists(template_file):
            self.slot_templet = pd.read_excel(template_file)
            self._process_slot_templates()
        else:
            # 创建默认槽位模板
            self._create_default_slot_templates()
    
    def _process_slot_templates(self):
        """处理槽位模板数据"""
        self.slot_info = {}
        for i in range(len(self.slot_templet)):
            slot = self.slot_templet.iloc[i]['slot']
            query = self.slot_templet.iloc[i]['query']
            values = self.slot_templet.iloc[i]['values']
            
            if slot not in self.slot_info:
                self.slot_info[slot] = {}
            self.slot_info[slot]['query'] = query
            self.slot_info[slot]['values'] = values
    
    def _create_default_slot_templates(self):
        """创建默认槽位模板"""
        self.slot_info = {
            "#服装类型#": {
                "query": "请问您想买什么类型的衣服？",
                "values": r"(上衣|裤子|裙子|外套|衬衫|T恤|牛仔裤|连衣裙)"
            },
            "#服装颜色#": {
                "query": "请问您喜欢什么颜色？",
                "values": r"(红色|蓝色|绿色|黄色|黑色|白色|灰色|粉色|紫色|橙色)"
            },
            "#服装尺寸#": {
                "query": "请问您需要什么尺寸？",
                "values": r"(S|M|L|XL|XXL|XS|XXXL)"
            },
            "#分期付款期数#": {
                "query": "请问您想分几期付款？",
                "values": r"(\d+期|\d+个月)"
            },
            "#支付方式#": {
                "query": "请问您使用什么支付方式？",
                "values": r"(支付宝|微信|银行卡|信用卡|现金)"
            }
        }


class BertIntentClassifier:
    """基于BERT的意图分类器"""
    
    def __init__(self, bert_model, tokenizer):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
    
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """获取文本的BERT嵌入向量"""
        inputs = self.tokenizer(text, return_tensors="pt", 
                               padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # 使用[CLS]标记的嵌入作为句子表示
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embedding[0]
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def classify_intent(self, query: str, available_nodes: List[str], 
                       all_node_info: Dict) -> Tuple[str, float]:
        """使用BERT进行意图分类"""
        query_embedding = self.get_bert_embedding(query)
        max_score = -1
        best_node = None
        
        for node_id in available_nodes:
            node_info = all_node_info[node_id]
            intents = node_info.get('intent', [])
            
            for intent in intents:
                intent_embedding = self.get_bert_embedding(intent)
                similarity = self.calculate_cosine_similarity(query_embedding, intent_embedding)
                
                if similarity > max_score:
                    max_score = similarity
                    best_node = node_id
        
        return best_node, max_score


class BertSlotFiller:
    """基于BERT的槽位填充器"""
    
    def __init__(self, bert_model, tokenizer):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
    
    def extract_slots_with_bert(self, query: str, slot_info: Dict) -> Dict[str, str]:
        """使用BERT进行槽位提取"""
        extracted_slots = {}
        
        for slot_name, slot_data in slot_info.items():
            # 使用正则表达式进行初步匹配
            pattern = slot_data['values']
            match = re.search(pattern, query)
            
            if match:
                extracted_slots[slot_name] = match.group()
            else:
                # 如果正则匹配失败，使用BERT进行语义匹配
                bert_match = self._bert_slot_matching(query, slot_name, slot_data)
                if bert_match:
                    extracted_slots[slot_name] = bert_match
        
        return extracted_slots
    
    def _bert_slot_matching(self, query: str, slot_name: str, slot_data: Dict) -> Optional[str]:
        """使用BERT进行语义槽位匹配"""
        # 这里可以实现更复杂的BERT-based槽位匹配逻辑
        # 目前先返回None，后续可以扩展
        return None


class RerunHandler:
    """重听功能处理器"""
    
    def __init__(self):
        self.conversation_history = []
    
    def add_turn(self, user_input: str, system_response: str, memory: Dict):
        """添加对话轮次到历史记录"""
        turn = {
            'user_input': user_input,
            'system_response': system_response,
            'memory': memory.copy()
        }
        self.conversation_history.append(turn)
    
    def handle_rerun_request(self, query: str) -> bool:
        """检测是否为重听请求"""
        rerun_keywords = ['重听', '再说一遍', '重复', '没听清', '什么']
        return any(keyword in query for keyword in rerun_keywords)
    
    def get_last_response(self) -> Optional[str]:
        """获取最后一次系统回复"""
        if self.conversation_history:
            return self.conversation_history[-1]['system_response']
        return None


class DialogueStateTracker:
    """对话状态跟踪器"""
    
    def __init__(self):
        self.current_state = {}
    
    def update_state(self, memory: Dict) -> Dict:
        """更新对话状态"""
        hit_node = memory.get("hit_node")
        if hit_node:
            node_info = memory.get("node_info", {})
            required_slots = node_info.get('slot', [])
            
            # 检查缺失的槽位
            missing_slots = []
            for slot in required_slots:
                if slot not in memory or not memory[slot]:
                    missing_slots.append(slot)
            
            memory["missing_slots"] = missing_slots
            memory["require_slot"] = missing_slots[0] if missing_slots else None
        
        return memory


class DialoguePolicyOptimizer:
    """对话策略优化器"""
    
    def __init__(self):
        self.policy_rules = {
            "reply": self._reply_policy,
            "request": self._request_policy,
            "rerun": self._rerun_policy
        }
    
    def optimize_policy(self, memory: Dict) -> Dict:
        """优化对话策略"""
        if memory.get("require_slot"):
            memory["policy"] = "request"
        elif memory.get("is_rerun"):
            memory["policy"] = "rerun"
        else:
            memory["policy"] = "reply"
        
        return self.policy_rules[memory["policy"]](memory)
    
    def _reply_policy(self, memory: Dict) -> Dict:
        """回复策略"""
        hit_node = memory["hit_node"]
        node_info = memory.get("node_info", {})
        memory["available_nodes"] = node_info.get("childnode", [])
        return memory
    
    def _request_policy(self, memory: Dict) -> Dict:
        """请求策略"""
        memory["available_nodes"] = [memory["hit_node"]]
        return memory
    
    def _rerun_policy(self, memory: Dict) -> Dict:
        """重听策略"""
        memory["available_nodes"] = memory.get("previous_available_nodes", [])
        return memory


class NaturalLanguageGenerator:
    """自然语言生成器"""
    
    def __init__(self, slot_info: Dict):
        self.slot_info = slot_info
    
    def generate_response(self, memory: Dict) -> str:
        """生成自然语言回复"""
        policy = memory.get("policy", "reply")
        
        if policy == "reply":
            return self._generate_reply_response(memory)
        elif policy == "request":
            return self._generate_request_response(memory)
        elif policy == "rerun":
            return self._generate_rerun_response(memory)
        else:
            return "抱歉，我没有理解您的意思。"
    
    def _generate_reply_response(self, memory: Dict) -> str:
        """生成回复响应"""
        node_info = memory.get("node_info", {})
        template = node_info.get("response", "好的，我明白了。")
        
        # 填充槽位
        for slot in node_info.get("slot", []):
            if slot in memory:
                template = template.replace(slot, memory[slot])
        
        return template
    
    def _generate_request_response(self, memory: Dict) -> str:
        """生成请求响应"""
        require_slot = memory.get("require_slot")
        if require_slot and require_slot in self.slot_info:
            return self.slot_info[require_slot]["query"]
        return "请提供更多信息。"
    
    def _generate_rerun_response(self, memory: Dict) -> str:
        """生成重听响应"""
        return memory.get("last_response", "抱歉，没有之前的回复可以重复。")


class BertMultiTurnDialogueSystem:
    """基于BERT的多轮对话系统主控制器"""
    
    def __init__(self, model_name: str = "/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese"):
        """初始化多轮对话系统"""
        self.dialogue_system = BertDialogueSystem(model_name)
        self.intent_classifier = BertIntentClassifier(
            self.dialogue_system.bert_model, 
            self.dialogue_system.tokenizer
        )
        self.slot_filler = BertSlotFiller(
            self.dialogue_system.bert_model, 
            self.dialogue_system.tokenizer
        )
        self.rerun_handler = RerunHandler()
        self.state_tracker = DialogueStateTracker()
        self.policy_optimizer = DialoguePolicyOptimizer()
        self.nlg = NaturalLanguageGenerator(self.dialogue_system.slot_info)
    
    def process_user_input(self, query: str, memory: Dict) -> Dict:
        """处理用户输入的主流程"""
        # 1. 检查重听请求
        if self.rerun_handler.handle_rerun_request(query):
            memory["is_rerun"] = True
            memory["last_response"] = self.rerun_handler.get_last_response()
        else:
            memory["is_rerun"] = False
            memory["query"] = query
        
        # 2. 意图识别
        if not memory.get("is_rerun"):
            memory = self._intent_recognition(memory)
        
        # 3. 槽位填充
        if not memory.get("is_rerun"):
            memory = self._slot_filling(memory)
        
        # 4. 对话状态跟踪
        memory = self.state_tracker.update_state(memory)
        
        # 5. 对话策略优化
        memory = self.policy_optimizer.optimize_policy(memory)
        
        # 6. 自然语言生成
        response = self.nlg.generate_response(memory)
        memory["response"] = response
        
        # 7. 更新历史记录
        self.rerun_handler.add_turn(query, response, memory)
        
        return memory
    
    def _intent_recognition(self, memory: Dict) -> Dict:
        """意图识别"""
        query = memory["query"]
        available_nodes = memory.get("available_nodes", [])
        
        if available_nodes:
            hit_node, score = self.intent_classifier.classify_intent(
                query, available_nodes, self.dialogue_system.all_node_info
            )
            memory["hit_node"] = hit_node
            memory["intent_score"] = score
            
            if hit_node:
                memory["node_info"] = self.dialogue_system.all_node_info[hit_node]
        
        return memory
    
    def _slot_filling(self, memory: Dict) -> Dict:
        """槽位填充"""
        query = memory["query"]
        node_info = memory.get("node_info", {})
        required_slots = node_info.get("slot", [])
        
        if required_slots:
            extracted_slots = self.slot_filler.extract_slots_with_bert(
                query, self.dialogue_system.slot_info
            )
            
            # 更新记忆中的槽位信息
            for slot, value in extracted_slots.items():
                if slot in required_slots:
                    memory[slot] = value
        
        return memory
    
    def start_conversation(self, initial_nodes: List[str]) -> Dict:
        """开始新对话"""
        memory = {
            "available_nodes": initial_nodes,
            "conversation_turns": 0
        }
        return memory


if __name__ == "__main__":
    # 测试用例
    print("正在初始化基于BERT的多轮对话系统...")
    
    try:
        # 创建多轮对话系统实例
        system = BertMultiTurnDialogueSystem()
   
        # 开始对话
        print("\n=== 开始对话测试 ===")
        memory = system.start_conversation([
            "scenario-买衣服_node1", 
            "scenario-看电影_node1"
        ])
        
        # 测试对话
        test_queries = [
            "我要买衣服",
            "买一件红色的T恤",
            "尺寸是L号",
            "重听",
            "可以分期付款吗"
        ]
        
        for query in test_queries:
            print(f"\n用户: {query}")
            memory = system.process_user_input(query, memory)
            print(f"系统: {memory['response']}")
            print(f"当前状态: {memory.get('policy', 'unknown')}")
        
    except Exception as e:
        print(f"系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
