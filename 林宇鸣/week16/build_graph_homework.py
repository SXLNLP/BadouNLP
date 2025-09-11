import re
import json
from py2neo import Graph
from collections import defaultdict

# 连接图数据库
graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}


# 清理实体名称和提取标签
def get_label_then_clean(x, label_data):
    # 去除特殊字符
    x = x.replace("《", "").replace("》", "")
    x = re.sub(r'[“”]', '', x)
    x = x.replace("、", "")  # 去掉中文逗号
    x = x.replace("&", "")  # 去掉&符号
    x = x.strip()  # 去掉两侧空格

    # 只替换非字母数字字符和特殊字符，但保留中文汉字
    x = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fa5]', '_', x)  # 用下划线替换不合法字符

    # 如果是纯数字或以数字开头，添加前缀以确保符合 Neo4j 的命名规则
    if x.isdigit() or (len(x) > 0 and x[0].isdigit()):
        x = "node_" + x  # 添加前缀

    return x


# 读取实体-关系-实体三元组文件
with open("triplets_head_rel_tail_GEM.txt", encoding="utf8") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")  # 取出三元组
        head = get_label_then_clean(head, label_data)
        tail = get_label_then_clean(tail, label_data)  # 也处理尾实体
        relation_data[head][relation] = tail

# 读取实体-属性名-属性值三元组文件
with open("triplets_enti_attr_value_GEM.txt", encoding="utf8") as f:
    for line in f:
        entity, attribute, value = line.strip().split("\t")  # 取出三元组
        entity = get_label_then_clean(entity, label_data)
        attribute_data[entity][attribute] = value.strip()  # 去掉值的两侧空格

# 构建 Cypher 语句
cypher = ""
in_graph_entity = set()

# 处理属性名-属性值的三元组
for entity in attribute_data:
    attribute_data[entity]["NAME"] = entity  # 为实体增加一个名字属性
    text = "{"
    for attribute, value in attribute_data[entity].items():
        text += "%s:'%s'," % (attribute, value)
    text = text[:-1] + "}"  # 最后一个逗号替换为大括号
    if entity in label_data:
        label = label_data[entity]
        cypher += "CREATE (%s:%s %s)\n" % (entity, label, text)  # 带标签的实体构建语句
    else:
        cypher += "CREATE (%s %s)\n" % (entity, text)  # 不带标签的实体构建语句
    in_graph_entity.add(entity)

# 处理关系三元组
for head in relation_data:
    if head not in in_graph_entity:
        cypher += "CREATE (%s {NAME:'%s'})\n" % (head, head)  # 为没有属性的实体增加名称属性
        in_graph_entity.add(head)

    for relation, tail in relation_data[head].items():
        if tail not in in_graph_entity:
            cypher += "CREATE (%s {NAME:'%s'})\n" % (tail, tail)  # 为没有属性的实体增加名称属性
            in_graph_entity.add(tail)

        cypher += "CREATE (%s)-[:%s]->(%s)\n" % (head, relation, tail)  # 关系语句

print(cypher)

# 执行建表脚本
graph.run(cypher)

# 记录我们图谱里都有哪些实体，哪些属性，哪些关系，哪些标签
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

# 修改文件名以保存新的 JSON 文件
with open("GEM_kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))


