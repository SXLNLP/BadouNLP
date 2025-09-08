from py2neo import Graph

# 连接Neo4j
graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))

print("=== Python函数知识图谱查询测试 ===\n")

# 测试查询
queries = [
    ("查看所有节点", "MATCH (n) RETURN n.name, labels(n) LIMIT 10"),
    ("len函数的属性", "MATCH (n) WHERE n.name = 'len' RETURN n"),
    ("len函数的应用场景", "MATCH (n)-[:USED_IN]->(m) WHERE n.name = 'len' RETURN m.name"),
    ("基础数据处理函数包含哪些", "MATCH (n)-[:CONTAINS]->(m) WHERE n.name = 'basic_data_functions' RETURN m.name"),
    ("高频使用的函数", "MATCH (n) WHERE n.frequency = 'very_high' OR n.frequency = 'high' RETURN n.name, n.frequency"),
    ("所有关系类型", "MATCH ()-[r]->() RETURN DISTINCT type(r)")
]

for desc, query in queries:
    print(f"查询: {desc}")
    try:
        result = graph.run(query).data()
        for item in result[:5]:  # 限制显示前5条
            print(f"  {item}")
        print()
    except Exception as e:
        print(f"  错误: {e}\n")

