import re
import json
import pandas
import itertools
from py2neo import Graph
from collections import defaultdict

'''
基于Python内置函数的知识图谱问答系统
'''

class PythonFunctionQA:
    def __init__(self):
        self.graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))
        schema_path = "kg_schema.json"
        templet_path = (r"D:\code\ai\week16\homework\question_templet.csv")
        self.load(schema_path, templet_path)
        print("Python函数知识图谱问答系统加载完毕！\n===============")

    def load(self, schema_path, templet_path):
        self.load_kg_schema(schema_path)
        self.load_question_templet(templet_path)
        return

    def load_kg_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
        self.relation_set = set(schema["relations"])
        self.entity_set = set(schema["entitys"])
        self.label_set = set(schema.get("labels", []))
        self.attribute_set = set(schema["attributes"])
        return

    def load_question_templet(self, templet_path):
        dataframe = pandas.read_csv(templet_path)
        self.question_templet = []
        for index in range(len(dataframe)):
            question = dataframe["question"][index]
            cypher = dataframe["cypher"][index]
            cypher_check = dataframe["check"][index]
            answer = dataframe["answer"][index]
            self.question_templet.append([question, cypher, json.loads(cypher_check), answer])
        return

    def get_mention_entitys(self, sentence):
        return re.findall("|".join(self.entity_set), sentence)

    def get_mention_relations(self, sentence):
        return re.findall("|".join(self.relation_set), sentence)

    def get_mention_attributes(self, sentence):
        return re.findall("|".join(self.attribute_set), sentence)

    def get_mention_labels(self, sentence):
        return re.findall("|".join(self.label_set), sentence)

    def parse_sentence(self, sentence):
        entitys = self.get_mention_entitys(sentence)
        relations = self.get_mention_relations(sentence)
        labels = self.get_mention_labels(sentence)
        attributes = self.get_mention_attributes(sentence)
        return {"%ENT%": entitys,
                "%REL%": relations,
                "%LAB%": labels,
                "%ATT%": attributes}

    def decode_value_combination(self, value_combination, cypher_check):
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
        slot_values = []
        for key, required_count in cypher_check.items():
            slot_values.append(itertools.combinations(info[key], required_count))
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
        return combinations

    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        combinations = self.get_combinations(cypher_check, info)
        templet_cpyher_pair = []
        for combination in combinations:
            replaced_templet = self.replace_token_in_string(templet, combination)
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            replaced_answer = self.replace_token_in_string(answer, combination)
            templet_cpyher_pair.append([replaced_templet, replaced_cypher, replaced_answer])
        return templet_cpyher_pair

    def check_cypher_info_valid(self, info, cypher_check):
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    def expand_question_and_cypher(self, info):
        templet_cypher_pair = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                templet_cypher_pair += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return templet_cypher_pair

    def sentence_similarity_function(self, string1, string2):
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance

    def cypher_match(self, sentence, info):
        templet_cypher_pair = self.expand_question_and_cypher(info)
        result = []
        for templet, cypher, answer in templet_cypher_pair:
            score = self.sentence_similarity_function(sentence, templet)
            result.append([templet, cypher, score, answer])
        result = sorted(result, reverse=True, key=lambda x: x[2])
        return result

    def parse_result(self, graph_search_result, answer, info):
        graph_search_result = graph_search_result[0]
        if "REL" in graph_search_result:
            graph_search_result["REL"] = list(graph_search_result["REL"].types())[0]
        answer = self.replace_token_in_string(answer, graph_search_result)
        return answer

    def query(self, sentence):
        print("============")
        print(sentence)
        info = self.parse_sentence(sentence)
        print("提取信息:", info)
        templet_cypher_score = self.cypher_match(sentence, info)
        for templet, cypher, score, answer in templet_cypher_score:
            print(f"匹配模板: {templet} (相似度: {score:.3f})")
            print(f"Cypher查询: {cypher}")
            graph_search_result = self.graph.run(cypher).data()
            if graph_search_result:
                answer = self.parse_result(graph_search_result, answer, info)
                return answer        
        return "未找到答案"

if __name__ == "__main__":
    graph = PythonFunctionQA()
    
    # 测试问题
    test_questions = [
        "len函数的作用是什么",
        "enumerate函数应用于什么场景",
        "哪些函数属于基础数据处理函数",
        "max函数的返回值是什么",
        "sum函数的使用频率如何"
    ]
    
    for question in test_questions:
        result = graph.query(question)
        print(f"答案: {result}\n")