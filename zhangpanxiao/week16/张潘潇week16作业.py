from neo4j import GraphDatabase

# Neo4j 数据库连接信息
URI = ""  # 默认连接协议和端口
AUTH = ("", "")   # 用户名和密码

# 定义查询函数
def run_query(query, params=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(query, params)
            return result.data()

# 示例问题：查询某人出演的电影
question = "Keanu Reeves出演了哪些电影？"
# 将自然语言问题映射到Cypher查询（这里是硬编码映射，真实系统需要NLP来解析）
cypher_query = """
    MATCH (p:Person {name: $name})-[:ACTED_IN]->(m:Movie)
    RETURN m.title AS answer
"""
query_params = {"name": "Keanu Reeves"}

# 执行查询并打印结果
answers = run_query(cypher_query, query_params)
print(f"问题: {question}")
for record in answers:
    print(f"- {record['answer']}")
