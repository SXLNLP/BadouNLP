import re
import json
from py2neo import Graph
from collections import defaultdict

# 连接图数据库
graph = Graph("http://localhost:7474", auth=("neo4j", "881201"))

attribute_data = defaultdict(dict)
# 修复：使用 list 支持一对多
relation_data = defaultdict(lambda: defaultdict(list))
label_data = {}

def get_label_then_clean(x):
    match = re.search(r"（(.+)）", x)
    if match:
        label_str = match.group(1)
        for label in ["歌曲", "专辑", "电影", "电视剧", "综艺"]:
            if label in label_str:
                clean_x = re.sub(r"（.+）", "", x).strip()
                label_data[x] = label
                return clean_x
        return re.sub(r"（.+）", "", x).strip()
    return x

# 读取 实体-关系-实体
with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        head, rel, tail = parts
        head = get_label_then_clean(head)
        tail = get_label_then_clean(tail)
        relation_data[head][rel].append(tail)  # ✅ 使用 append

# 读取 属性
with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        entity, attr, val = parts
        entity = get_label_then_clean(entity)
        attribute_data[entity][attr] = val

# =============================
# 构建 Cypher
# =============================

cypher_lines = []
# =============================
# Step 1: 收集所有实体
# =============================
all_entities = set()
all_entities.update(attribute_data.keys())

for head, rel_dict in relation_data.items():
    all_entities.add(head)
    for tails in rel_dict.values():
        all_entities.update(tails)

# 分配变量名
entity_to_var = {entity: f"n{i}" for i, entity in enumerate(all_entities)}

# =============================
# Step 2: 为所有实体创建节点（无论是否有属性）
# =============================
for entity in all_entities:
    var = entity_to_var[entity]
    if entity in attribute_data:
        props = attribute_data[entity].copy()
        props["NAME"] = entity
        props = {k: v.replace("'", "\\'") for k, v in props.items()}
        prop_str = ", ".join(f"{k}:'{v}'" for k, v in props.items())
        if entity in label_data:
            label = label_data[entity]
            cypher_lines.append(f"CREATE ({var}:`{label}` {{{prop_str}}})")
        else:
            cypher_lines.append(f"CREATE ({var}:Entity {{{prop_str}}})")
    else:
        safe_entity = entity.replace("'", "\\'")
        cypher_lines.append(f"CREATE ({var}:Entity {{NAME:'{safe_entity}'}})")

# 创建只有关系的实体
for head in relation_data:
    if head not in entity_to_var:
        continue
    h_var = entity_to_var[head]
    for relation, tails in relation_data[head].items():
        for tail in tails:
            if tail not in entity_to_var:
                safe_tail = tail.replace("'", "\\'")
                t_var = entity_to_var[tail]
                cypher_lines.append(f"CREATE ({t_var}:Entity {{NAME:'{safe_tail}'}})")
            r_name = relation.replace("'", "\\'")
            t_var = entity_to_var[tail]
            cypher_lines.append(f"CREATE ({h_var})-[:`{r_name}`]->({t_var})")

# 执行
cypher = "\n".join(cypher_lines)
print("Generated Cypher:\n", cypher)

try:
    graph.run(cypher)
    print("✅ 知识图谱构建成功！")
except Exception as e:
    print("❌ 执行失败：", str(e))
# 可以打印前几行调试
    print("\n前几行 Cypher：")
    print("\n".join(cypher.split("\n")[:10]))
# =============================
# 输出 schema
# =============================
data = defaultdict(set)

# 1. 从关系三元组中提取实体和关系
for head in relation_data:
    data["entitys"].add(head)  # 添加头实体
    for relation, tails in relation_data[head].items():  # tails 是列表
        data["relations"].add(relation)
        for tail in tails:  # ✅ 遍历每个尾实体
            data["entitys"].add(tail)

# 2. 从标签中提取 labels
for enti, label in label_data.items():
    data["labels"].add(label)

# 3. 从属性三元组中提取实体和属性
for enti in attribute_data:
    data["entitys"].add(enti)
    for attr in attribute_data[enti]:
        data["attributes"].add(attr)

# 转为字典，保存
data = {k: list(v) for k, v in data.items()}  # set 自动去重

with open("kg_schema.json", "w", encoding="utf8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("📊 schema 已保存到 kg_schema.json")