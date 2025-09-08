#基本bert和知识图谱和问题模版的 问答
import re
import json
import torch
import numpy as np
import pandas as pd
import itertools
from transformers import BertTokenizer, BertModel
from py2neo import Graph
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from collections import defaultdict

"""
基于question_templet.xlsx模板，使用BERT增强实体识别
"""

class SimpleBertGraphQA:
    def __init__(self, neo4j_uri="http://localhost:7474", neo4j_user="neo4j", neo4j_password="neo4j", 
                 bert_model_path="bert-base-chinese"):    
        
        # 初始化Neo4j连接
        self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # 初始化BERT模型（仅用于实体识别）
        print("正在加载BERT模型...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path)
        self.bert_model.eval()
        
        # 加载模板和schema
        schema_path = "kg_schema.json"
        templet_path = "question_templet.xlsx"
        self.load(schema_path, templet_path)
        
        # 构建实体BERT嵌入
        self.entity_embeddings = {}
        self.build_entity_embeddings()
        
        print("简化版BERT+知识图谱问答系统加载完毕！\n===============")

    def load(self, schema_path, templet_path):
        """加载模板和schema"""
        self.load_kg_schema(schema_path)
        self.load_question_templet(templet_path)

    def load_kg_schema(self, path):
        """加载图谱信息"""
        try:
            with open(path, encoding="utf8") as f:
                schema = json.load(f)
            self.relation_set = set(schema.get("relations", []))
            self.entity_set = set(schema.get("entitys", []))
            self.label_set = set(schema.get("labels", []))
            self.attribute_set = set(schema.get("attributes", []))
            
            print(f"加载知识图谱schema: {len(self.entity_set)}个实体, {len(self.relation_set)}个关系")
            
        except FileNotFoundError:
            print(f"未找到{path}文件")
            self.relation_set = set()
            self.entity_set = set()
            self.label_set = set()
            self.attribute_set = set()

    def load_question_templet(self, templet_path):
        """加载模板信息"""
        try:
            dataframe = pd.read_excel(templet_path)
            self.question_templet = []
            for index in range(len(dataframe)):
                question = dataframe["question"][index]
                cypher = dataframe["cypher"][index]
                cypher_check = dataframe["check"][index]
                answer = dataframe["answer"][index]
                self.question_templet.append([question, cypher, json.loads(cypher_check), answer])
            
            print(f"加载问题模板: {len(self.question_templet)}个模板")
            
        except FileNotFoundError:
            print(f"未找到{templet_path}文件")
            self.question_templet = []
        except Exception as e:
            print(f"加载模板出错: {e}")
            self.question_templet = []

    def get_bert_embedding(self, text):
        """获取文本的BERT嵌入"""
        if not text or len(text.strip()) == 0:
            return np.zeros(768)
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True, 
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state[0][0].numpy()
        
        return embedding

    def build_entity_embeddings(self):
        """为所有实体构建BERT嵌入"""
        print("构建实体BERT嵌入...")
        
        entity_count = 0
        for entity in self.entity_set:
            if entity and len(entity.strip()) > 0:
                embedding = self.get_bert_embedding(entity)
                self.entity_embeddings[entity] = embedding
                entity_count += 1
                
                if entity_count % 100 == 0:
                    print(f"已处理 {entity_count} 个实体嵌入")
        
        print(f"完成 {len(self.entity_embeddings)} 个实体的BERT嵌入构建")

    def get_mention_entitys_bert(self, sentence):
        """使用BERT增强的实体识别"""
        
        # 方法1：精确匹配（保留原有逻辑）
        exact_entities = re.findall("|".join(self.entity_set), sentence)
        
        if exact_entities:
            return exact_entities
        
        # 方法2：BERT语义匹配
        sentence_embedding = self.get_bert_embedding(sentence)
        
        similarities = []
        for entity, entity_embedding in self.entity_embeddings.items():
            similarity = cosine_similarity([sentence_embedding], [entity_embedding])[0][0]
            if similarity > 0.7:  # 相似度阈值
                similarities.append((entity, similarity))
        
        # 按相似度排序，返回最相似的实体
        similarities.sort(key=lambda x: x[1], reverse=True)
        bert_entities = [entity for entity, sim in similarities[:3]]
        
        # 方法3：分词后匹配
        words = list(jieba.cut(sentence))
        word_entities = []
        
        for word in words:
            if len(word) > 1:
                for entity in self.entity_set:
                    if word in entity or entity in word:
                        word_entities.append(entity)
        
        # 合并所有结果并去重
        all_entities = list(set(exact_entities + bert_entities + word_entities))
        return all_entities[:5]  # 返回前5个

    def get_mention_relations(self, sentence):
        """获取问题中谈到的关系（保持原有逻辑）"""
        return re.findall("|".join(self.relation_set), sentence)

    def get_mention_attributes(self, sentence):
        """获取问题中谈到的属性（保持原有逻辑）"""
        return re.findall("|".join(self.attribute_set), sentence)

    def get_mention_labels(self, sentence):
        """获取问题中谈到的标签（保持原有逻辑）"""
        return re.findall("|".join(self.label_set), sentence)

    def parse_sentence(self, sentence):
        """对问题进行预处理，提取需要的信息（使用BERT增强的实体识别）"""
        entitys = self.get_mention_entitys_bert(sentence)  # 使用BERT增强版本
        relations = self.get_mention_relations(sentence)
        labels = self.get_mention_labels(sentence)
        attributes = self.get_mention_attributes(sentence)
        return {"%ENT%": entitys,
                "%REL%": relations,
                "%LAB%": labels,
                "%ATT%": attributes}

    def decode_value_combination(self, value_combination, cypher_check):
        """将提取到的值分配到键上（保持原有逻辑）"""
        res = {}
        for index, (key, required_count) in enumerate(cypher_check.items()):
            if required_count == 1:
                res[key] = value_combination[index][0]
            else:
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    res[key_num] = value_combination[index][i]
        return res

    def get_combinations(self, cypher_check, info):
        """对于找到了超过模板中需求的实体数量的情况，需要进行排列组合（保持原有逻辑）"""
        slot_values = []
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) >= required_count:
                slot_values.append(itertools.combinations(info[key], required_count))
            else:
                return []  # 如果某个槽位的值不够，返回空列表
        
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
        return combinations

    def replace_token_in_string(self, string, combination):
        """将带有token的模板替换成真实词（保持原有逻辑）"""
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        """对于单条模板，根据抽取到的实体属性信息扩展（保持原有逻辑）"""
        combinations = self.get_combinations(cypher_check, info)
        templet_cypher_pair = []
        for combination in combinations:
            replaced_templet = self.replace_token_in_string(templet, combination)
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            replaced_answer = self.replace_token_in_string(answer, combination)
            templet_cypher_pair.append([replaced_templet, replaced_cypher, replaced_answer])
        return templet_cypher_pair

    def check_cypher_info_valid(self, info, cypher_check):
        """验证从文本中提取到的信息是否足够填充模板（保持原有逻辑）"""
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    def expand_question_and_cypher(self, info):
        """根据提取到的实体，关系等信息，将模板展开成待匹配的问题文本（保持原有逻辑）"""
        templet_cypher_pair = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                templet_cypher_pair += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return templet_cypher_pair

    def sentence_similarity_function(self, string1, string2):
        """距离函数，使用BERT计算语义相似度"""
        
        # 原有的Jaccard距离作为基础分数
        jaccard_score = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        
        # 使用BERT计算语义相似度
        try:
            embedding1 = self.get_bert_embedding(string1)
            embedding2 = self.get_bert_embedding(string2)
            bert_similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # 组合两种相似度分数：70% BERT + 30% Jaccard
            combined_score = 0.7 * bert_similarity + 0.3 * jaccard_score
            return combined_score
            
        except Exception as e:
            print(f"BERT相似度计算出错: {e}")
            return jaccard_score

    def cypher_match(self, sentence, info):
        """通过问题匹配的方式确定匹配的cypher（使用BERT增强相似度计算）"""
        templet_cypher_pair = self.expand_question_and_cypher(info)
        result = []
        for templet, cypher, answer in templet_cypher_pair:
            score = self.sentence_similarity_function(sentence, templet)
            result.append([templet, cypher, score, answer])
        result = sorted(result, reverse=True, key=lambda x: x[2])
        return result

    def parse_result(self, graph_search_result, answer, info):
        """解析结果（保持原有逻辑）"""
        if not graph_search_result:
            return None
            
        graph_search_result = graph_search_result[0]
        
        # 关系查找返回的结果形式较为特殊，单独处理
        if "REL" in graph_search_result:
            graph_search_result["REL"] = list(graph_search_result["REL"].types())[0]
        
        answer = self.replace_token_in_string(answer, graph_search_result)
        return answer

    def query(self, sentence):
        """对外提供问答接口（保持原有逻辑，但使用BERT增强的实体识别）"""
        print("============")
        print(f"问题: {sentence}")
        
        # 使用BERT增强的信息抽取
        info = self.parse_sentence(sentence)
        print(f"提取的信息: {info}")
        
        # cypher匹配（使用BERT增强的相似度计算）
        templet_cypher_score = self.cypher_match(sentence, info)
        
        if not templet_cypher_score:
            return "无法匹配到合适的查询模板。"
        
        # 尝试执行查询，找到第一个有结果的模板
        for templet, cypher, score, answer in templet_cypher_score:
            print(f"尝试模板: {templet} (相似度: {score:.3f})")
            print(f"生成查询: {cypher}")
            
            try:
                graph_search_result = self.graph.run(cypher).data()
                
                # 找到答案时停止查找后面的模板
                if graph_search_result:
                    final_answer = self.parse_result(graph_search_result, answer, info)
                    print(f"查询结果: {graph_search_result}")
                    return final_answer
                else:
                    print("该查询无结果，尝试下一个模板...")
                    
            except Exception as e:
                print(f"查询执行出错: {e}")
                continue
        
        return "在知识图谱中没有找到相关信息。"

    def build_entity_embeddings(self):
        """为实体构建BERT嵌入"""
        print("构建实体BERT嵌入...")
        
        entity_count = 0
        for entity in self.entity_set:
            if entity and len(entity.strip()) > 0:
                embedding = self.get_bert_embedding(entity)
                self.entity_embeddings[entity] = embedding
                entity_count += 1
                
                if entity_count % 100 == 0:
                    print(f"已处理 {entity_count} 个实体嵌入")
        
        print(f"完成 {len(self.entity_embeddings)} 个实体的BERT嵌入构建")

    def batch_test(self):
        """批量测试（基于原有测试用例）"""
        test_questions = [
            "谁导演的不能说的秘密",
            "发如雪的谱曲是谁", 
            "爱在西元前的谱曲是谁",
            "周杰伦的星座是什么",
            "周杰伦的血型是什么",
            "周杰伦的身高",
            "周杰伦和淡江中学是什么关系",
            "周杰伦是什么星座的？",  # 测试BERT语义匹配
            "杰伦的血型？",          # 测试部分匹配
            "谁写的发如雪？"         # 测试同义词匹配
        ]
        
        print("=== 批量测试 ===\n")
        
        for question in test_questions:
            try:
                answer = self.query(question)
                print(f"最终答案: {answer}")
                print("=" * 60)
            except Exception as e:
                print(f"处理问题出错: {e}")
                print("=" * 60)

    def interactive_qa(self):
        """交互式问答模式"""
        print("\n=== 交互式问答模式 ===")
        print("基于BERT+模板的知识图谱问答系统")
        print("输入问题，输入 'quit' 退出\n")
        
        while True:
            question = input("请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出', 'q']:
                break
            
            if not question:
                continue
            
            try:
                answer = self.query(question)
                print(f"答案: {answer}")
                print("-" * 40)
            except Exception as e:
                print(f"处理问题时出错: {e}")

def main():
    """主函数"""
    print("=== 简化版BERT+知识图谱问答系统 ===\n")
    
    try:
        # 初始化系统
        qa_system = SimpleBertGraphQA(
            neo4j_uri="http://localhost:7474",
            neo4j_user="neo4j", 
            neo4j_password="demo",  # 请替换为你的Neo4j密码
            bert_model_path="bert-base-chinese"
        )
        
        # 检查系统状态
        if not qa_system.entity_set:
            print("警告: 未加载到实体数据，请先运行build_graph.py构建知识图谱")
            return
        
        if not qa_system.question_templet:
            print("警告: 未加载到问题模板，请检查question_templet.xlsx文件")
            return
        
        # 批量测试
        qa_system.batch_test()
        
        # 交互式问答
        qa_system.interactive_qa()
        
    except Exception as e:
        print(f"系统初始化失败: {e}")
        print("请检查:")
        print("1. Neo4j是否正在运行")
        print("2. 文件是否存在: kg_schema.json, question_templet.xlsx")
        print("3. 是否已安装依赖: pip install transformers torch scikit-learn py2neo pandas jieba")

if __name__ == "__main__":
    main()
