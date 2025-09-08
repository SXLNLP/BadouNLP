import re
import json
from py2neo import Graph
from collections import defaultdict

\
#连接图数据库
graph = Graph("http://localhost:7474",auth=("neo4j","demo"))

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)
label_data = {}


# 提取到标签后，把括号部分删除
def get_label_then_clean(x, label_data):
    if re.search("（.+）", x):
        label_string = re.search("（.+）", x).group()
        for label in ["歌曲", "专辑", "电影", "电视剧"]:
            if label in label_string:
                x = re.sub("（.+）", "", x)  # 括号内的内容删掉，因为括号是特殊字符会影响cypher语句运行 # 有的实体后面有括号，里面的内容可以作为标签
                label_data[x] = label
            else:
                x = re.sub("（.+）", "", x)
    return x

# 读取all_triplets.txt文件 实体-关系-实体 
print("开始读取all_triplets.txt文件...")
with open("all_triplets.txt", encoding="utf8") as f:
    line_count = 0
    for line in f:
        try:
            head, relation, tail = line.strip().split("\t")  #取出三元组
            head = get_label_then_clean(head, label_data)
            relation_data[head][relation] = tail
            line_count += 1
            
            if line_count % 100 == 0:
                print(f"已处理 {line_count} 个关系三元组")
                
        except Exception as e:
            print(f"处理行出错: {line.strip()} - {e}")
            continue

print(f"完成读取 {line_count} 个关系三元组")

#读取实体-属性-属性值三元组文件 
print("开始读取triplets_enti_attr_value.txt文件...")
with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    attr_count = 0
    for line in f:
        try:
            entity, attribute, value = line.strip().split("\t")  # 取出三元组
            entity = get_label_then_clean(entity, label_data)
            attribute_data[entity][attribute] = value
            attr_count += 1
            
        except Exception as e:
            print(f"处理属性行出错: {line.strip()} - {e}")
            continue

print(f"完成读取 {attr_count} 个属性三元组")

#构建cypher语句
print("开始构建Cypher语句...")
cypher = ""
in_graph_entity = set()

for i, entity in enumerate(attribute_data):
    #为所有的实体增加一个名字属性
    attribute_data[entity]["NAME"] = entity
    #将一个实体的所有属性拼接成一个类似字典的表达式
    text = "{"
    for attribute, value in attribute_data[entity].items():
        # 转义特殊字符
        escaped_value = value.replace("'", "\\'").replace('"', '\\"')
        text += "%s:'%s',"%(attribute, escaped_value)
    text = text[:-1] + "}"  #最后一个逗号替换为大括号
    
    if entity in label_data:
        label = label_data[entity]
        #带标签的实体构建语句
        cypher += "CREATE (%s:%s %s)" % (entity, label, text) + "\n"
    else:
        #不带标签的实体构建语句
        cypher += "CREATE (%s:Entity %s)" % (entity, text) + "\n"
    in_graph_entity.add(entity)

#构建关系语句
for i, head in enumerate(relation_data):
    # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
    if head not in in_graph_entity:
        cypher += "CREATE (%s:Entity {NAME:'%s'})" % (head, head) + "\n"
        in_graph_entity.add(head)

    for relation, tail in relation_data[head].items():
        # 有可能实体只有和其他实体的关系，但没有属性，为这样的实体增加一个名称属性，便于在图上认出
        if tail not in in_graph_entity:
            cypher += "CREATE (%s:Entity {NAME:'%s'})" % (tail, tail) + "\n"
            in_graph_entity.add(tail)

        #关系语句
        cypher += "CREATE (%s)-[:%s]->(%s)" % (head, relation, tail) + "\n"

print(f"Cypher语句构建完成，准备创建 {len(in_graph_entity)} 个实体")

# 清空现有数据
print("清空现有图谱数据...")
graph.run("MATCH (n) DETACH DELETE n")

#执行建表脚本
try:
    graph.run(cypher)
    print("图谱构建成功！")
except Exception as e:
    print(f"执行Cypher出错: {e}")
    print("尝试分批执行...")
    
    # 分批执行避免大量数据导致的问题
    cypher_lines = [line for line in cypher.strip().split('\n') if line.strip()]
    batch_size = 50
    success_count = 0
    
    for i in range(0, len(cypher_lines), batch_size):
        batch = cypher_lines[i:i+batch_size]
        batch_cypher = '\n'.join(batch)
        
        try:
            graph.run(batch_cypher)
            success_count += len(batch)
            print(f"已成功执行 {success_count}/{len(cypher_lines)} 条语句")
        except Exception as batch_error:
            print(f"批次执行出错: {batch_error}")


print("生成图谱schema...")
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

# 输出统计信息
print(f"\n图谱构建统计:")
print(f"  实体数量: {len(data.get('entitys', []))}")
print(f"  属性数量: {len(data.get('attributes', []))}")  
print(f"  关系数量: {len(data.get('relations', []))}")
print(f"  标签数量: {len(data.get('labels', []))}")

with open("kg_schema.json", "w", encoding="utf8") as f:
    f.write(json.dumps(data, ensure_ascii=False, indent=2))

print("Schema文件已保存到 kg_schema.json")
print("图谱构建完成！")
