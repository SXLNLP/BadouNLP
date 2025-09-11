import re
import json
import pandas as pd
import itertools
from py2neo import Graph
from collections import defaultdict


class BasketballGraphQA:
    def __init__(self):
        # Neo4j 连接配置（需与数据库一致）
        self.graph = Graph("http://localhost:7474", auth=("neo4j", "123"))
        # 图谱 schema 文件（导入篮球数据后生成的结构文件）
        self.schema_path = r"N:\八斗\八斗精品班\第十六周 知识图谱\week16 知识图谱问答\kgqa_base_on_sentence_match\homework\basketball_kg_schema.json"
        # 问题模板 Excel 文件
        self.templet_path = r"N:\八斗\八斗精品班\第十六周 知识图谱\week16 知识图谱问答\kgqa_base_on_sentence_match\homework\template.xlsx"
        # self.templet_path = r"N:\八斗\八斗精品班\第十六周 知识图谱\week16 知识图谱问答\kgqa_base_on_sentence_match\homework\question_templet.xlsx"
        self.load()
        print("篮球知识图谱问答系统加载完毕！\n===============")

    def load(self):
        self.load_kg_schema(self.schema_path)
        self.load_question_templet(self.templet_path)

    def load_kg_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
        self.relation_set = set(schema["relations"])
        self.entity_set = set(schema["entities"])
        self.label_set = set(schema["labels"])
        self.attribute_set = set(schema["attributes"])

    def load_question_templet(self, templet_path):
        df = pd.read_excel(templet_path)
        self.question_templet = []
        for idx in range(len(df)):
            question = str(df["question"][idx]).strip()
            cypher = str(df["cypher"][idx]).strip()
            cypher_check = str(df["check"][idx]).strip()
            answer = str(df["answer"][idx]).strip()
            try:
                cypher_check = json.loads(cypher_check.replace("'", '"'))
            except:
                print(f"模板 {idx+1} 解析失败，跳过")
                continue
            self.question_templet.append([question, cypher, cypher_check, answer])

    def get_mention_entitys(self, sentence):
        if not self.entity_set:
            return []
        sorted_ents = sorted(self.entity_set, key=len, reverse=True)
        pattern = "|".join(re.escape(ent) for ent in sorted_ents)
        return list(set(re.findall(pattern, sentence)))

    def get_mention_relations(self, sentence):
        if not self.relation_set:
            return []
        sorted_rels = sorted(self.relation_set, key=len, reverse=True)
        pattern = "|".join(re.escape(rel) for rel in sorted_rels)
        return list(set(re.findall(pattern, sentence)))

    def get_mention_attributes(self, sentence):
        if not self.attribute_set:
            return []
        sorted_attrs = sorted(self.attribute_set, key=len, reverse=True)
        pattern = "|".join(re.escape(attr) for attr in sorted_attrs)
        return list(set(re.findall(pattern, sentence)))

    def get_mention_values(self, sentence):
        return re.findall(r"\d+[a-zA-Z]*", sentence)

    def get_mention_labels(self, sentence):
        """从问题中提取标签（如“篮球运动员”“球队”）"""
        if not self.label_set:
            return []
        sorted_labs = sorted(self.label_set, key=lambda x: len(x), reverse=True)
        pattern = "|".join(re.escape(lab) for lab in sorted_labs)
        return list(set(re.findall(pattern, sentence)))


    def parse_sentence(self, sentence):
        info = {
            "%ENT%": self.get_mention_entitys(sentence),
            "%REL%": self.get_mention_relations(sentence),
            "%LAB%": self.get_mention_labels(sentence),
            "%ATT%": self.get_mention_attributes(sentence),
            "%VAL%": self.get_mention_values(sentence)
        }
        print("提取的信息：", info)
        return info

    def decode_value_combination(self, value_combination, cypher_check):
        res = {}
        for idx, (key, req_count) in enumerate(cypher_check.items()):
            if req_count == 1:
                res[key] = value_combination[idx][0]
            else:
                for i in range(req_count):
                    key_num = f"{key[:-1]}{i}%"
                    res[key_num] = value_combination[idx][i]
        return res

    def get_combinations(self, cypher_check, info):
        slot_values = []
        for key, req_count in cypher_check.items():
            slot_values.append(itertools.combinations(info.get(key, []), req_count))
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for combo in value_combinations:
            combinations.append(self.decode_value_combination(combo, cypher_check))
        return combinations

    def replace_token_in_string(self, string, combination):
        for key, val in combination.items():
            string = string.replace(key, str(val))
        return string

    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        combinations = self.get_combinations(cypher_check, info)
        expanded_pairs = []
        for combo in combinations:
            replaced_templet = self.replace_token_in_string(templet, combo)
            replaced_cypher = self.replace_token_in_string(cypher, combo)
            replaced_answer = self.replace_token_in_string(answer, combo)
            expanded_pairs.append([replaced_templet, replaced_cypher, replaced_answer])
        return expanded_pairs

    def check_cypher_info_valid(self, info, cypher_check):
        for key, req_count in cypher_check.items():
            if len(info.get(key, [])) < req_count:
                return False
        return True

    def expand_question_and_cypher(self, info):
        expanded_pairs = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                expanded_pairs += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return expanded_pairs

    def sentence_similarity_function(self, s1, s2):
        """基于词的相似度：计算问题与模板的词重叠率"""
        words1 = re.findall(r'[\u4e00-\u9fa5]+|\w+', s1)  # 提取中文/英文/数字词
        words2 = re.findall(r'[\u4e00-\u9fa5]+|\w+', s2)
        common_words = set(words1) & set(words2)
        return len(common_words) / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0

    def cypher_match(self, sentence, info):
        expanded_pairs = self.expand_question_and_cypher(info)
        matches = []
        for templet, cypher, answer in expanded_pairs:
            score = self.sentence_similarity_function(sentence, templet)
            matches.append([templet, cypher, score, answer])
        matches.sort(reverse=True, key=lambda x: x[2])
        return matches

    def parse_result(self, graph_result, answer, info):
        if not graph_result:
            return "未找到相关信息~"
        if len(graph_result) > 1:
            values = [res[list(res.keys())[0]] for res in graph_result]
            return answer.replace("{" + list(graph_result[0].keys())[0] + "}", "、".join(values))
        first_res = graph_result[0]
        if "REL" in first_res:
            first_res["REL"] = list(first_res["REL"].types())[0]
        return self.replace_token_in_string(answer, first_res)

    def query(self, sentence):
        print("============")
        print("问题：", sentence)
        info = self.parse_sentence(sentence)
        matches = self.cypher_match(sentence, info)
        if not matches:
            return "未找到匹配的问题模板~"
        for templet, cypher, score, answer in matches[:3]:
            print(f"匹配的模板（相似度：{score:.2f}）：", templet)
            print("执行的 Cypher：", cypher)
            try:
                graph_res = self.graph.run(cypher).data()
                if graph_res:
                    print("图查询结果：", graph_res)
                    return self.parse_result(graph_res, answer, info)
            except Exception as e:
                print(f"Cypher执行错误：{e}")
        return "未找到答案，请尝试其他问题~"


if __name__ == "__main__":
    bball_qa = BasketballGraphQA()
    test_questions = [
        "詹姆斯的身高是多少？",
        "库里的所属球队是哪个？",
        "杜兰特和詹姆斯是什么关系？",
        "谁的总冠军数是4？",
        "詹姆斯的经纪人是谁？",
        "库里的曾效力球队有哪些？"
    ]
    for q in test_questions:
        answer = bball_qa.query(q)
        print("答案：", answer, "\n")