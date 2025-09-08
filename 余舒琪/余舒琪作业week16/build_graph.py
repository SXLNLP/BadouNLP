import json
from py2neo import Graph
from collections import defaultdict

'''
读取三元组，并将数据写入neo4j
'''


#连接图数据库
graph = Graph("http://localhost:7474",auth=("neo4j","neo4j977"))


attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}


#读取实体-关系-实体三元组文件
with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        head, relation, tail = line.strip().split(" - ")  #取出三元组
        label_data[head] = "ENTERPRISE"
        label_data[tail] = "ENTERPRISE"
        relation_data[head][relation] = tail


#读取实体-属性-属性值三元组文件
with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        entity, attribute, value = line.strip().split(" - ")  # 取出三元组
        label_data[entity] = "ENTERPRISE"
        attribute_data[entity][attribute] = value


#构建cypher语句
cypher = ""
in_graph_entity = set()
for i, entity in enumerate(attribute_data):
    #为所有的实体增加一个名字属性
    attribute_data[entity]["NAME"] = entity
    #将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in attribute_data[entity].items():
        text += "%s:\'%s\',"%(attribute, value)
    text = text[:-1] + "}"  #最后一个逗号替换为大括号
    label = label_data[entity]
    #带标签的实体构建语句
    cypher += "CREATE (%s:%s %s)" % (entity, label, text) + "\n"
    in_graph_entity.add(entity)

#构建关系语句
for i, head in enumerate(relation_data):

    # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
    if head not in in_graph_entity:
        label = label_data[head]
        cypher += "CREATE (%s:%s {NAME:'%s'})" % (head, label, head) + "\n"
        in_graph_entity.add(head)

    for relation, tail in relation_data[head].items():

        # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
        if tail not in in_graph_entity:
            label = label_data[tail]
            cypher += "CREATE (%s:%s {NAME:'%s'})" % (tail, label, tail) + "\n"
            in_graph_entity.add(tail)

        #关系语句
        cypher += "CREATE (%s)-[:%s]->(%s)" % (head, relation, tail) + "\n"

print(cypher)

#执行建表脚本
graph.run(cypher)

#记录我们图谱里都有哪些实体，哪些属性，哪些关系，哪些标签
data = defaultdict(set)
for head in relation_data:
    data["entities"].add(head)
    for relation, tail in relation_data[head].items():
        data["relations"].add(relation)
        data["entities"].add(tail)

for label in label_data.values():
    data["labels"].add(label)

for enti in attribute_data:
    for attr, value in attribute_data[enti].items():
        data["entities"].add(enti)
        data["attributes"].add(attr)

data = dict((x, list(y)) for x, y in data.items())

with open("kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))
