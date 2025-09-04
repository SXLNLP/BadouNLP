import re
import json
from py2neo import Graph
from collections import defaultdict

'''
读取三元组，并将数据写入neo4j
'''

# 连接图数据库
graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}
entity_to_var = {}  # 初始化实体到变量名的映射字典


# 有的实体后面有括号，里面的内容可以作为标签
# 提取到标签后，把括号部分删除
def get_label_then_clean(x, label_data):
    if re.search("（.+）", x):
        label_string = re.search("（.+）", x).group()
        for label in ["歌曲", "专辑", "电影", "电视剧"]:
            if label in label_string:
                x = re.sub("（.+）", "", x)  # 括号内的内容删掉，因为括号是特殊字符会影响cypher语句运行
                label_data[x] = label
            else:
                x = re.sub("（.+）", "", x)
    return x


# 读取实体-关系-实体三元组文件
with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")  # 取出三元组
        head = get_label_then_clean(head, label_data)
        relation_data[head][relation] = tail

# 读取实体-属性-属性值三元组文件
with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        entity, attribute, value = line.strip().split("\t")  # 取出三元组
        entity = get_label_then_clean(entity, label_data)
        attribute_data[entity][attribute] = value

# 构建cypher语句
cypher = ""
in_graph_entity = set()
node_counter = 0  # 节点计数器，用于生成唯一的变量名

# 首先处理有属性的实体
for entity in attribute_data:
    # 为节点生成合法的变量名
    node_var = f"n{node_counter}"
    node_counter += 1

    # 为所有的实体增加一个名字属性
    attribute_data[entity]["NAME"] = entity

    # 将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in attribute_data[entity].items():
        # 转义单引号，防止属性值中包含单引号导致语法错误
        escaped_value = str(value).replace("'", "\\'")
        text += f"{attribute}:'{escaped_value}',"
    text = text[:-1] + "}"

    if entity in label_data:
        label = label_data[entity]
        # 带标签的实体构建语句 - 使用变量名而不是实体名
        cypher += f"CREATE ({node_var}:{label} {text})" + "\n"
    else:
        # 不带标签的实体构建语句
        cypher += f"CREATE ({node_var} {text})" + "\n"

    in_graph_entity.add(entity)
    # 记录变量名与实体名的映射
    entity_to_var[entity] = node_var

# 处理只有关系没有属性的实体
for head in relation_data:
    if head not in in_graph_entity:
        head_var = f"n{node_counter}"
        node_counter += 1
        cypher += f"CREATE ({head_var} {{NAME:'{head}'}})" + "\n"
        in_graph_entity.add(head)
        entity_to_var[head] = head_var

    for relation, tail in relation_data[head].items():
        if tail not in in_graph_entity:
            tail_var = f"n{node_counter}"
            node_counter += 1
            cypher += f"CREATE ({tail_var} {{NAME:'{tail}'}})" + "\n"
            in_graph_entity.add(tail)
            entity_to_var[tail] = tail_var

# 构建关系
for head in relation_data:
    for relation, tail in relation_data[head].items():
        head_var = entity_to_var.get(head)
        tail_var = entity_to_var.get(tail)

        if head_var and tail_var:
            # 清理关系名称中的特殊字符，只保留字母、数字和下划线
            clean_relation = re.sub(r'[^a-zA-Z0-9_]', '_', relation)
            cypher += f"CREATE ({head_var})-[:{clean_relation}]->({tail_var})" + "\n"

print("生成的Cypher语句长度:", len(cypher))
# 查看生成的Cypher语句
# print(cypher[:1000] + "..." if len(cypher) > 1000 else cypher)

# 分批执行建表脚本
try:
    # 将Cypher语句按行分割，分批执行
    cypher_lines = cypher.strip().split('\n')
    batch_size = 100  # 每批执行100条语句

    for i in range(0, len(cypher_lines), batch_size):
        batch = cypher_lines[i:i + batch_size]
        batch_cypher = '\n'.join(batch)
        graph.run(batch_cypher)
        print(f"已执行第 {i // batch_size + 1} 批，共 {len(batch)} 条语句")

    print("数据导入完成！")

except Exception as e:
    print(f"执行过程中出现错误: {e}")
    print("请检查生成的Cypher语句是否有语法错误")

# 记录我们图谱里都有哪些实体，哪些属性，哪些关系，哪些标签
data = defaultdict(set)
for head in relation_data:
    data["entities"].add(head)
    for relation, tail in relation_data[head].items():
        data["relations"].add(relation)
        data["entities"].add(tail)

for enti, label in label_data.items():
    data["entities"].add(enti)
    data["labels"].add(label)

for enti in attribute_data:
    for attr, value in attribute_data[enti].items():
        data["entities"].add(enti)
        data["attributes"].add(attr)

data = dict((x, list(y)) for x, y in data.items())

with open("kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))

print("图谱模式已保存到 kg_schema.json")
