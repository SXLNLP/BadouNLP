import re
import json
from py2neo import Graph
from collections import defaultdict

'''
读取三元组，并将数据写入neo4j
'''

# 连接图数据库
try:
    graph = Graph("http://localhost:7474", auth=("neo4j", "polo-input-sharp-protect-people-3644"))
    print("成功连接到Neo4j数据库")
except Exception as e:
    print(f"连接Neo4j数据库失败: {e}")
    graph = None

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}


# 清理节点名称，确保符合Cypher语法
def clean_node_name(name):
    # 移除特殊字符，只保留字母、数字、下划线
    cleaned = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', str(name))
    # 如果以数字开头，在前面加下划线
    if cleaned and cleaned[0].isdigit():
        cleaned = '_' + cleaned
    # 确保名称不为空
    if not cleaned:
        cleaned = 'node_' + str(hash(name))[:8]
    return cleaned


# 有的实体后面有括号，里面的内容可以作为标签
# 提取到标签后，把括号部分删除
def get_label_then_clean(x, label_data):
    if re.search("（.+）", x):
        label_string = re.search("（.+）", x).group()
        for label in ["公司", "产品", "技术", "服务", "人物", "地点", "智能手机", "笔记本电脑", "平板电脑", "智能手表",
                      "无线耳机", "总部园区", "零售店", "编程语言", "高科技公司"]:
            if label in label_string:
                x = re.sub("（.+）", "", x)  # 括号内的内容删掉，因为括号是特殊字符会影响cypher语句运行
                label_data[x] = label
                break
        else:
            x = re.sub("（.+）", "", x)
    return x.strip()


# 读取实体-属性-属性值三元组文件 (apple_1.txt)
try:
    with open("apple_1.txt", encoding="utf8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 使用正则表达式分割
            parts = re.split(r'[\s\u3000\u2002]+', line)

            if len(parts) >= 3:
                entity = parts[0]
                attribute = parts[1]
                value = ' '.join(parts[2:])

                entity = get_label_then_clean(entity, label_data)
                attribute_data[entity][attribute] = value

except Exception as e:
    print(f"处理apple_1.txt时出错: {e}")

# 读取实体-关系-实体三元组文件 (apple_2.txt)
try:
    with open("apple_2.txt", encoding="utf8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 使用正则表达式分割
            parts = re.split(r'[\s\u3000\u2002]+', line)

            if len(parts) >= 3:
                head = parts[0]
                relation = parts[1]
                tail = ' '.join(parts[2:])

                head = get_label_then_clean(head, label_data)
                tail = get_label_then_clean(tail, label_data)
                relation_data[head][relation] = tail

except Exception as e:
    print(f"处理apple_2.txt时出错: {e}")

# 调试信息：显示读取的数据
print(f"\n=== 读取的数据统计 ===")
print(f"属性数据条目: {len(attribute_data)}")
for entity, attrs in attribute_data.items():
    print(f"  {entity}: {attrs}")
print(f"关系数据条目: {len(relation_data)}")
for head, rels in relation_data.items():
    for rel, tail in rels.items():
        print(f"  {head} -{rel}-> {tail}")
print(f"标签数据: {label_data}")

# 构建cypher语句
cypher = ""
in_graph_entity = set()
node_mapping = {}  # 映射原始名称到清理后的名称

# 首先处理所有实体及其属性
for i, entity in enumerate(attribute_data):
    clean_entity = clean_node_name(entity)
    node_mapping[entity] = clean_entity

    # 为所有的实体增加一个名字属性
    attribute_data[entity]["NAME"] = entity
    # 将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in attribute_data[entity].items():
        # 转义单引号，避免cypher语句错误
        value = str(value).replace("'", "\\'")
        text += "%s:'%s'," % (attribute, value)
    text = text[:-1] + "}"  # 最后一个逗号替换为大括号

    if entity in label_data:
        label = label_data[entity]
        # 带标签的实体构建语句
        cypher += "CREATE (%s:%s %s)" % (clean_entity, label, text) + "\n"
    else:
        # 不带标签的实体构建语句
        cypher += "CREATE (%s %s)" % (clean_entity, text) + "\n"
    in_graph_entity.add(entity)

# 构建关系语句
for i, head in enumerate(relation_data):
    # 清理头部实体名称
    clean_head = clean_node_name(head)
    node_mapping[head] = clean_head

    # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性
    if head not in in_graph_entity:
        cypher += "CREATE (%s {NAME:'%s'})" % (clean_head, head) + "\n"
        in_graph_entity.add(head)

    for relation, tail in relation_data[head].items():
        # 清理尾部实体名称
        clean_tail = clean_node_name(tail)
        node_mapping[tail] = clean_tail

        # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性
        if tail not in in_graph_entity:
            cypher += "CREATE (%s {NAME:'%s'})" % (clean_tail, tail) + "\n"
            in_graph_entity.add(tail)

        # 清理关系名称
        clean_relation = clean_node_name(relation)

        # 关系语句（使用清理后的实体名称）
        cypher += "CREATE (%s)-[:%s]->(%s)" % (clean_head, clean_relation, clean_tail) + "\n"

print("\n=== 生成的Cypher语句 ===")
if cypher:
    print(cypher[:500] + "..." if len(cypher) > 500 else cypher)
else:
    print("没有生成Cypher语句")

# 执行建表脚本
if graph and cypher.strip():
    try:
        result = graph.run(cypher)
        print("数据成功导入Neo4j!")
    except Exception as e:
        print(f"导入数据到Neo4j时出错: {e}")
        # 如果有错误，尝试逐条执行
        print("尝试逐条执行Cypher语句...")
        for statement in cypher.split('\n'):
            if statement.strip():
                try:
                    graph.run(statement.strip())
                except Exception as inner_e:
                    print(f"执行失败: {statement.strip()} - 错误: {inner_e}")
else:
    print("未连接到数据库或没有Cypher语句，跳过执行")

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
    data["entitys"].add(enti)
    for attr, value in attribute_data[enti].items():
        data["attributes"].add(attr)

data = dict((x, list(y)) for x, y in data.items())

try:
    with open("apple_kg_schema.json", "w", encoding="utf8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
    print("知识图谱模式已保存到 apple_kg_schema.json")
except Exception as e:
    print(f"保存schema文件时出错: {e}")

print("程序执行完成!")
