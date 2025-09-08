import re
import json
from py2neo import Graph
from collections import defaultdict

'''
从TXT文件读取篮球运动员实体和关系数据（空格分隔），并将数据写入neo4j
'''

# 连接图数据库
# 请根据你的Neo4j配置修改以下参数
graph = Graph("http://localhost:7474", auth=("neo4j", "123"))

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}

# 设置实体标签
label_data["詹姆斯"] = "篮球运动员"
label_data["库里"] = "篮球运动员"
label_data["杜兰特"] = "篮球运动员"
label_data["洛杉矶湖人"] = "球队"
label_data["金州勇士"] = "球队"
label_data["菲尼克斯太阳"] = "球队"
label_data["克里夫兰骑士"] = "球队"
label_data["莱昂内尔·霍林斯沃斯"] = "经纪人"
label_data["萨肖尔·科恩"] = "经纪人"
label_data["拉尔夫·劳伦"] = "经纪人"

relation_file_path = r"N:\八斗\八斗精品班\第十六周 知识图谱\week16 知识图谱问答\kgqa_base_on_sentence_match\homework\player_relation.txt"
attribute_file_path = r"N:\八斗\八斗精品班\第十六周 知识图谱\week16 知识图谱问答\kgqa_base_on_sentence_match\homework\player_entity_attr.txt"

try:
    with open(attribute_file_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 按任意数量的空格分割每行数据（处理多个空格的情况）
            parts = re.split(r'\s+', line)
            if len(parts) == 3:
                entity, attribute, value = parts
                attribute_data[entity][attribute] = value
            else:
                print(f"警告：无效的属性数据格式 - {line}")
except FileNotFoundError:
    print(f"错误：找不到实体属性文件 {attribute_file_path}")
    exit(1)
except Exception as e:
    print(f"读取实体属性文件时出错：{str(e)}")
    exit(1)

try:
    with open(relation_file_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            # 按任意数量的空格分割每行数据（处理多个空格的情况）
            parts = re.split(r'\s+', line)
            if len(parts) == 3:
                head, relation, tail = parts
                # 处理一对多关系
                if relation not in relation_data[head]:
                    relation_data[head][relation] = []
                relation_data[head][relation].append(tail)
            else:
                print(f"警告：无效的关系数据格式 - {line}")
except FileNotFoundError:
    print(f"错误：找不到关系文件 {relation_file_path}")
    exit(1)
except Exception as e:
    print(f"读取关系文件时出错：{str(e)}")
    exit(1)

# 构建cypher语句
cypher = ""
in_graph_entity = set()

# 创建实体及其属性
for entity in attribute_data:
    # 为所有的实体增加一个名字属性
    attribute_data[entity]["NAME"] = entity
    # 将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in attribute_data[entity].items():
        # 处理数值类型的属性，去掉引号
        if str(value).isdigit():
            text += f"{attribute}:{value},"
        else:
            text += f"{attribute}:'{value}',"
    text = text[:-1] + "}"  # 最后一个逗号替换为大括号
    
    if entity in label_data:
        label = label_data[entity]
        # 带标签的实体构建语句，使用反引号处理可能的特殊字符
        cypher += f"CREATE (`{entity}`:{label} {text})" + "\n"
    else:
        # 不带标签的实体构建语句
        cypher += f"CREATE (`{entity}` {text})" + "\n"
    in_graph_entity.add(entity)

# 构建关系语句
for head in relation_data:
    # 处理可能只有关系没有属性的实体
    if head not in in_graph_entity:
        cypher += f"CREATE (`{head}` {{NAME:'{head}'}})" + "\n"
        in_graph_entity.add(head)
    
    for relation, tails in relation_data[head].items():
        for tail in tails:
            # 处理可能只有关系没有属性的实体
            if tail not in in_graph_entity:
                cypher += f"CREATE (`{tail}` {{NAME:'{tail}'}})" + "\n"
                in_graph_entity.add(tail)
            
            # 关系语句，使用反引号处理可能的特殊字符
            cypher += f"CREATE (`{head}`)-[:{relation}]->(`{tail}`)" + "\n"

print("生成的Cypher语句:\n", cypher)

# 执行建表脚本
graph.run(cypher)

# 记录图谱里的实体、属性、关系和标签
data = defaultdict(set)
for head in relation_data:
    data["entities"].add(head)
    for relation, tails in relation_data[head].items():
        data["relations"].add(relation)
        for tail in tails:
            data["entities"].add(tail)

for entity, label in label_data.items():
    data["entities"].add(entity)
    data["labels"].add(label)

for entity in attribute_data:
    for attr, value in attribute_data[entity].items():
        data["entities"].add(entity)
        data["attributes"].add(attr)

# 转换为字典并保存
data = dict((x, list(y)) for x, y in data.items())

with open("basketball_kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))

print("数据已成功导入Neo4j，图谱结构已保存到basketball_kg_schema.json")
