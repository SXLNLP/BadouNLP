import re
import json
from py2neo import Graph
from collections import defaultdict

'''
基于Python内置函数构建知识图谱
'''

#连接图数据库 - 请根据你的Neo4j配置修改
graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}

# 提取标签的函数
def get_label_then_clean(x, label_data):
    if re.search("（.+）", x):
        label_string = re.search("（.+）", x).group()
        for label in ["函数", "方法", "操作", "应用"]:
            if label in label_string:
                x = re.sub("（.+）", "", x)
                label_data[x] = label
            else:
                x = re.sub("（.+）", "", x)
    return x

#读取实体-关系-实体三元组文件
with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")
        head = get_label_then_clean(head, label_data)
        relation_data[head][relation] = tail

#读取实体-属性-属性值三元组文件  
with open(r"d:\code\ai\week16\homework\triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        entity, attribute, value = line.strip().split("\t")
        entity = get_label_then_clean(entity, label_data)
        attribute_data[entity][attribute] = value

# 安全的节点名称处理函数
def safe_node_name(name):
    # 替换特殊字符，确保Cypher语法正确
    safe_name = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', name)
    return safe_name

#构建cypher语句
cypher = ""
in_graph_entity = set()
for i, entity in enumerate(attribute_data):
    #为所有的实体增加一个名字属性
    attribute_data[entity]["NAME"] = entity
    safe_entity = safe_node_name(entity)
    
    #将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in attribute_data[entity].items():
        safe_attr = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', attribute)
        safe_value = value.replace("'", "\\'")
        text += "%s:'%s'," % (safe_attr, safe_value)
    text = text[:-1] + "}"
    
    if entity in label_data:
        label = label_data[entity]
        cypher += "CREATE (%s:%s %s)" % (safe_entity, label, text) + "\n"
    else:
        cypher += "CREATE (%s %s)" % (safe_entity, text) + "\n"
    in_graph_entity.add(entity)

#构建关系语句
for i, head in enumerate(relation_data):
    safe_head = safe_node_name(head)
    
    if head not in in_graph_entity:
        cypher += "CREATE (%s {NAME:'%s'})" % (safe_head, head.replace("'", "\\'")) + "\n"
        in_graph_entity.add(head)

    for relation, tail in relation_data[head].items():
        safe_tail = safe_node_name(tail)
        safe_relation = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', relation)
        
        if tail not in in_graph_entity:
            cypher += "CREATE (%s {NAME:'%s'})" % (safe_tail, tail.replace("'", "\\'")) + "\n"
            in_graph_entity.add(tail)
        #关系语句
        cypher += "CREATE (%s)-[:%s]->(%s)" % (safe_head, safe_relation, safe_tail) + "\n"

print(cypher)

#执行建表脚本
graph.run(cypher)

#记录图谱的schema信息
data = defaultdict(set)
for head in relation_data:
    data["entitys"].add(head)
    for relation, tail in relation_data[head].items():
        data["relations"].add(relation)
        data["entitys"].add(tail)

for enti, label in label_data.items():
    data["entitys"].add(enti)
    data["labels"].add(label)

for enti in attribute_data:
    for attr, value in attribute_data[enti].items():
        data["entitys"].add(enti)
        data["attributes"].add(attr)

data = dict((x, list(y)) for x, y in data.items())

with open("kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))

print("Python函数知识图谱构建完成！")