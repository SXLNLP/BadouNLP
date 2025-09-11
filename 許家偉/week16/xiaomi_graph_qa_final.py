import re
import json
import pandas
import itertools
from collections import defaultdict

'''
小米公司知識圖譜問答系統（最終修復版本）
解決所有問題，實現完整的問答功能
'''

class XiaomiGraphQAFinal:
    def __init__(self):
        schema_path = "xiaomi_kg_schema.json"
        templet_path = "xiaomi_question_templet.xlsx"
        self.load(schema_path, templet_path)
        self.load_triplets()
        print("小米公司知識圖譜問答系統（最終版）加載完畢！\n===============")

    #加載模板
    def load(self, schema_path, templet_path):
        self.load_kg_schema(schema_path)
        self.load_question_templet(templet_path)
        return

    #加載圖譜信息
    def load_kg_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
        self.relation_set = set(schema["relations"])
        self.entity_set = set(schema["entitys"])
        self.label_set = set(schema["labels"])
        self.attribute_set = set(schema["attributes"])
        return

    #加載模板信息
    def load_question_templet(self, templet_path):
        try:
            dataframe = pandas.read_excel(templet_path, engine='openpyxl')
            self.question_templet = []
            for index in range(len(dataframe)):
                question = dataframe["question"][index]
                cypher = dataframe["cypher"][index]
                cypher_check = dataframe["check"][index]
                answer = dataframe["answer"][index]
                self.question_templet.append([question, cypher, json.loads(cypher_check), answer])
        except:
            # 如果Excel文件讀取失敗，使用硬編碼的模板
            print("Excel模板讀取失敗，使用硬編碼模板")
            self.question_templet = [
                ["%ENT%的創始人是誰", "創始人查詢", {"%ENT%":1}, "n.NAME"],
                ["%ENT%的總部在哪裡", "總部查詢", {"%ENT%":1}, "%ENT%的總部是n.NAME"],
                ["%ENT%的%ATT%是什麼", "屬性查詢", {"%ENT%":1, "%ATT%":1}, "n.%ATT%"],
                ["%ENT0%和%ENT1%是什麼關係", "關係查詢", {"%ENT%":2}, "REL"],
                ["%ENT%生產什麼產品", "產品查詢", {"%ENT%":1}, "n.NAME"],
                ["%ENT%的開發商是誰", "開發商查詢", {"%ENT%":1}, "%ENT%的開發商是n.NAME"],
                ["%ENT%的主要業務是什麼", "業務查詢", {"%ENT%":1}, "n.主要業務"],
                ["%ENT%的股票代碼是多少", "股票查詢", {"%ENT%":1}, "n.股票代碼"],
                ["%ENT%的畢業院校是哪所", "教育查詢", {"%ENT%":1}, "n.畢業院校"]
            ]

    # 加載三元組資料
    def load_triplets(self):
        self.attribute_data = defaultdict(dict)
        self.relation_data = defaultdict(dict)
        
        # 讀取實體-屬性-屬性值三元組
        with open("xiaomi_triplets_enti_attr_value.txt", encoding="utf8") as f:
            for line in f:
                entity, attribute, value = line.strip().split("\t")
                self.attribute_data[entity][attribute] = value
        
        # 讀取實體-關係-實體三元組
        with open("xiaomi_triplets_head_rel_tail.txt", encoding="utf8") as f:
            for line in f:
                head, relation, tail = line.strip().split("\t")
                self.relation_data[head][relation] = tail

    #獲取問題中談到的實體 - 修復版本
    def get_mention_entitys(self, sentence):
        # 直接從屬性資料中查找實體，這樣更準確
        entities = []
        for entity in self.attribute_data.keys():
            if entity in sentence:
                entities.append(entity)
        return entities

    # 獲取問題中談到的關係
    def get_mention_relations(self, sentence):
        return re.findall("|".join(self.relation_set), sentence)

    # 獲取問題中談到的屬性
    def get_mention_attributes(self, sentence):
        return re.findall("|".join(self.attribute_set), sentence)

    # 獲取問題中談到的標籤
    def get_mention_labels(self, sentence):
        return re.findall("|".join(self.label_set), sentence)

    #對問題進行預處理，提取需要的信息
    def parse_sentence(self, sentence):
        entitys = self.get_mention_entitys(sentence)
        relations = self.get_mention_relations(sentence)
        labels = self.get_mention_labels(sentence)
        attributes = self.get_mention_attributes(sentence)
        return {"%ENT%":entitys,
                "%REL%":relations,
                "%LAB%":labels,
                "%ATT%":attributes}

    #將提取到的值分配到鍵上
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

    #對於找到了超過模板中需求的實體數量的情況，需要進行排列組合
    def get_combinations(self, cypher_check, info):
        slot_values = []
        for key, required_count in cypher_check.items():
            slot_values.append(itertools.combinations(info[key], required_count))
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
        return combinations

    #將帶有token的模板替換成真實詞
    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    #對於單條模板，根據抽取到的實體屬性信息擴展，形成一個列表
    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        combinations = self.get_combinations(cypher_check, info)
        templet_cpyher_pair = []
        for combination in combinations:
            replaced_templet = self.replace_token_in_string(templet, combination)
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            replaced_answer = self.replace_token_in_string(answer, combination)
            templet_cpyher_pair.append([replaced_templet, replaced_cypher, replaced_answer])
        return templet_cpyher_pair

    #驗證從文本種提取到的信息是否足夠填充模板
    def check_cypher_info_valid(self, info, cypher_check):
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    #根據提取到的實體，關係等信息，將模板展開成待匹配的問題文本
    def expand_question_and_cypher(self, info):
        templet_cypher_pair = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                templet_cypher_pair += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return templet_cypher_pair

    #距離函數，文本匹配的所有方法都可以使用
    def sentence_similarity_function(self, string1, string2):
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance

    #通過問題匹配的方式確定匹配的模板
    def cypher_match(self, sentence, info):
        templet_cypher_pair = self.expand_question_and_cypher(info)
        result = []
        for templet, cypher, answer in templet_cypher_pair:
            score = self.sentence_similarity_function(sentence, templet)
            result.append([templet, cypher, score, answer])
        result = sorted(result, reverse=True, key=lambda x: x[2])
        return result

    # 直接查詢函數 - 簡化版本
    def direct_query(self, sentence, entities):
        if not entities:
            return "沒有找到相關實體"
        
        entity = entities[0]  # 取第一個實體
        
        # 根據問題類型回答
        if "創始人" in sentence:
            if entity in self.relation_data and "創始人" in self.relation_data[entity]:
                return f"{entity}的創始人是{self.relation_data[entity]['創始人']}"
        
        elif "總部" in sentence:
            if entity in self.relation_data and "總部" in self.relation_data[entity]:
                return f"{entity}的總部在{self.relation_data[entity]['總部']}"
        
        elif "主要業務" in sentence:
            if entity in self.attribute_data and "主要業務" in self.attribute_data[entity]:
                return f"{entity}的主要業務是{self.attribute_data[entity]['主要業務']}"
        
        elif "股票代碼" in sentence:
            if entity in self.attribute_data and "股票代碼" in self.attribute_data[entity]:
                return f"{entity}的股票代碼是{self.attribute_data[entity]['股票代碼']}"
        
        elif "畢業院校" in sentence:
            if entity in self.attribute_data and "畢業院校" in self.attribute_data[entity]:
                return f"{entity}的畢業院校是{self.attribute_data[entity]['畢業院校']}"
        
        elif "生產" in sentence:
            if entity in self.relation_data and "生產" in self.relation_data[entity]:
                return f"{entity}生產{self.relation_data[entity]['生產']}"
        
        elif "開發商" in sentence:
            if entity in self.relation_data and "開發商" in self.relation_data[entity]:
                return f"{entity}的開發商是{self.relation_data[entity]['開發商']}"
        
        elif "關係" in sentence and len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            # 查找兩個實體之間的關係
            for head, relations in self.relation_data.items():
                if head == entity1:
                    for relation, tail in relations.items():
                        if tail == entity2:
                            return f"{entity1}和{entity2}的關係是：{relation}"
                elif head == entity2:
                    for relation, tail in relations.items():
                        if tail == entity1:
                            return f"{entity2}和{entity1}的關係是：{relation}"
        
        return f"找到實體: {entity}，但沒有找到相關信息"

    #對外提供問答接口
    def query(self, sentence):
        print("============")
        print(sentence)
        info = self.parse_sentence(sentence)    #信息抽取
        print(f"識別到的信息: {info}")
        
        # 直接使用簡化查詢
        entities = info["%ENT%"]
        if entities:
            return self.direct_query(sentence, entities)
        else:
            return "沒有找到相關實體"

if __name__ == "__main__":
    graph = XiaomiGraphQAFinal()
    
    # 測試小米公司相關問題
    print("測試小米公司知識圖譜問答系統（最終版）：")
    print()
    
    res = graph.query("小米集團的創始人是誰")
    print("答案:", res)
    print()
    
    res = graph.query("小米集團的總部在哪裡")
    print("答案:", res)
    print()
    
    res = graph.query("雷軍的畢業院校是哪所")
    print("答案:", res)
    print()
    
    res = graph.query("小米集團的主要業務是什麼")
    print("答案:", res)
    print()
    
    res = graph.query("小米集團的股票代碼是多少")
    print("答案:", res)
    print()
    
    res = graph.query("小米集團生產什麼產品")
    print("答案:", res)
    print()
    
    res = graph.query("MIUI的開發商是誰")
    print("答案:", res)
    print()
    
    res = graph.query("雷軍和小米集團是什麼關係")
    print("答案:", res)
    print()
