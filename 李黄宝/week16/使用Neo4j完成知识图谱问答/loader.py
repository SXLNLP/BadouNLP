import re
import json
from py2neo import Graph
from collections import defaultdict

'''
# 数据处理 - 读取三元组，并将数据写入neo4j
'''
zhuti_data = defaultdict(dict)
shuxing_data = defaultdict(dict)
label_data = {}

#连接图数据库
graph = Graph("http://localhost:7474",auth=("neo4j","bado"))
#清空数据库
graph.run("MATCH (n) DETACH DELETE n")


#读取实体-关系-实体三元组文件
with open("sjz_zhuti.txt", encoding="utf8") as f:
    print(f)
    for line in f:
        print(line)
        head, relation, tail = line.strip().split("\t")  #取出三元组
        zhuti_data[head][relation] = tail


print("实体-关系-实体三元组文件读取完毕", zhuti_data)

#读取实体-属性-属性值三元组文件
with open("sjz_shuxing.txt", encoding="utf8") as f:
    for line in f:
        entity, attribute, value = line.strip().split("\t")  # 取出三元组
        shuxing_data[entity][attribute] = value

#构建cypher语句
cypher = ""
in_graph_entity = set()
for i, entity in enumerate(shuxing_data):
    #为所有的实体增加一个名字属性
    shuxing_data[entity]["NAME"] = entity
    #将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in shuxing_data[entity].items():
        text += "%s:\'%s\',"%(attribute, value)
    text = text[:-1] + "}"  #最后一个逗号替换为大括号
    #不带标签的实体构建语句
    cypher += "CREATE (%s:%s %s)" % (entity, entity, text) + "\n"
    # cypher += "CREATE (%s %s)" % (entity, text) + "\n"
    in_graph_entity.add(entity)
#构建关系语句
for i, head in enumerate(zhuti_data):
    # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
    if head not in in_graph_entity:
        cypher += "CREATE (%s {NAME:'%s'})" % (head, head) + "\n"
        in_graph_entity.add(head)

    for relation, tail in zhuti_data[head].items():

        # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
        if tail not in in_graph_entity:
            cypher += "CREATE (%s {NAME:'%s'})" % (tail, tail) + "\n"
            in_graph_entity.add(tail)

        #关系语句
        cypher += "CREATE (%s)-[:%s]->(%s)" % (head, relation, tail) + "\n"

print(cypher)

#执行建表脚本
graph.run(cypher)


#记录我们图谱里都有哪些实体，哪些属性，哪些关系，哪些标签
data = defaultdict(set)
for head in shuxing_data:
    data["entitys"].add(head)
    for relation, tail in shuxing_data[head].items():
        data["relations"].add(relation)
        data["entitys"].add(tail)

for enti, label in label_data.items():
    data["entitys"].add(enti)

for enti in zhuti_data:
    for attr, value in zhuti_data[enti].items():
        data["entitys"].add(enti)
        data["attributes"].add(attr)

data = dict((x, list(y)) for x, y in data.items())

with open("kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))