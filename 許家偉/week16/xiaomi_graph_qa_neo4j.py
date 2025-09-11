import re
import json
import pandas
import itertools
from py2neo import Graph
from collections import defaultdict

'''
小米公司知識圖譜問答系統 - Neo4j版本
連接Neo4j數據庫進行查詢
'''

class XiaomiGraphQANeo4j:
    def __init__(self):
        self.graph = self.connect_neo4j()
        if not self.graph:
            print("❌ 無法連接到Neo4j，退出")
            return
            
        schema_path = "xiaomi_kg_schema.json"
        templet_path = "xiaomi_question_templet.xlsx"
        self.load(schema_path, templet_path)
        print("小米公司知識圖譜問答系統（Neo4j版）加載完畢！\n===============")

    def connect_neo4j(self):
        """連接到Neo4j數據庫"""
        try:
            graph = Graph("neo4j://localhost:7687", auth=("neo4j", "admin852"))
            # 測試連接
            result = graph.run("RETURN 1 as test").data()
            print("✅ 成功連接到Neo4j數據庫！")
            return graph
        except Exception as e:
            print(f"❌ 連接Neo4j失敗: {e}")
            print("請確保Neo4j服務正在運行，並且連接參數正確")
            return None

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

    #獲取問題中談到的實體
    def get_mention_entitys(self, sentence):
        entities = []
        for entity in self.entity_set:
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

    # 使用Neo4j進行查詢
    def neo4j_query(self, sentence, entities):
        """使用Neo4j查詢知識圖譜"""
        if not entities:
            return "沒有找到相關實體"
        
        entity = entities[0]  # 取第一個實體
        
        # 根據問題類型構建Cypher查詢
        if "創始人" in sentence:
            cypher = f"""
            MATCH (n)-[:創始人]->(m)
            WHERE n.NAME = '{entity}'
            RETURN m.NAME as result
            """
            result = self.graph.run(cypher).data()
            if result:
                return f"{entity}的創始人是{result[0]['result']}"
        
        elif "總部" in sentence:
            cypher = f"""
            MATCH (n)-[:總部]->(m)
            WHERE n.NAME = '{entity}'
            RETURN m.NAME as result
            """
            result = self.graph.run(cypher).data()
            if result:
                return f"{entity}的總部在{result[0]['result']}"
        
        elif "主要業務" in sentence:
            cypher = f"""
            MATCH (n)
            WHERE n.NAME = '{entity}'
            RETURN n.主要業務 as result
            """
            result = self.graph.run(cypher).data()
            if result and result[0]['result']:
                return f"{entity}的主要業務是{result[0]['result']}"
        
        elif "股票代碼" in sentence:
            cypher = f"""
            MATCH (n)
            WHERE n.NAME = '{entity}'
            RETURN n.股票代碼 as result
            """
            result = self.graph.run(cypher).data()
            if result and result[0]['result']:
                return f"{entity}的股票代碼是{result[0]['result']}"
        
        elif "畢業院校" in sentence:
            cypher = f"""
            MATCH (n)
            WHERE n.NAME = '{entity}'
            RETURN n.畢業院校 as result
            """
            result = self.graph.run(cypher).data()
            if result and result[0]['result']:
                return f"{entity}的畢業院校是{result[0]['result']}"
        
        elif "生產" in sentence:
            cypher = f"""
            MATCH (n)-[:生產]->(m)
            WHERE n.NAME = '{entity}'
            RETURN m.NAME as result
            """
            result = self.graph.run(cypher).data()
            if result:
                products = [r['result'] for r in result]
                return f"{entity}生產的產品包括：{', '.join(products)}"
        
        elif "開發商" in sentence:
            cypher = f"""
            MATCH (n)-[:開發商]->(m)
            WHERE n.NAME = '{entity}'
            RETURN m.NAME as result
            """
            result = self.graph.run(cypher).data()
            if result:
                return f"{entity}的開發商是{result[0]['result']}"
        
        elif "關係" in sentence and len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            # 查找兩個實體之間的關係
            cypher = f"""
            MATCH (n)-[r]->(m)
            WHERE n.NAME = '{entity1}' AND m.NAME = '{entity2}'
            RETURN type(r) as result
            """
            result = self.graph.run(cypher).data()
            if result:
                return f"{entity1}和{entity2}的關係是：{result[0]['result']}"
            
            # 反向查找
            cypher = f"""
            MATCH (n)-[r]->(m)
            WHERE n.NAME = '{entity2}' AND m.NAME = '{entity1}'
            RETURN type(r) as result
            """
            result = self.graph.run(cypher).data()
            if result:
                return f"{entity2}和{entity1}的關係是：{result[0]['result']}"
        
        return f"找到實體: {entity}，但沒有找到相關信息"

    #對外提供問答接口
    def query(self, sentence):
        print("============")
        print(sentence)
        info = self.parse_sentence(sentence)    #信息抽取
        print(f"識別到的信息: {info}")
        
        # 使用Neo4j查詢
        entities = info["%ENT%"]
        if entities:
            return self.neo4j_query(sentence, entities)
        else:
            return "沒有找到相關實體"

if __name__ == "__main__":
    qa = XiaomiGraphQANeo4j()
    
    if not qa.graph:
        print("系統初始化失敗")
        exit(1)
    
    # 測試小米公司相關問題
    print("測試小米公司知識圖譜問答系統（Neo4j版）：")
    print()
    
    res = qa.query("小米集團的創始人是誰")
    print("答案:", res)
    print()
    
    res = qa.query("小米集團的總部在哪裡")
    print("答案:", res)
    print()
    
    res = qa.query("雷軍的畢業院校是哪所")
    print("答案:", res)
    print()
    
    res = qa.query("小米集團的主要業務是什麼")
    print("答案:", res)
    print()
    
    res = qa.query("小米集團的股票代碼是多少")
    print("答案:", res)
    print()
    
    res = qa.query("小米集團生產什麼產品")
    print("答案:", res)
    print()
    
    res = qa.query("MIUI的開發商是誰")
    print("答案:", res)
    print()
    
    res = qa.query("雷軍和小米集團是什麼關係")
    print("答案:", res)
    print()
