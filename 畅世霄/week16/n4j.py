from neo4j import GraphDatabase

# Neo4j 数据库连接信息
URI = ""  # 连接协议和端口
AUTH = ("", "")   # 用户名和密码

# 定义查询函数
def run_query(query, params=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(query, params)
            return result.data()

# 查询麦弗逊式悬架影响哪些性能指标
question = "麦弗逊式悬架影响哪些性能指标？"
# 将自然语言问题映射到Cypher查询
cypher_query = """
    MATCH (sus:SuspensionSystem {suspension_type: $type})<-[:DEPENDS_ON]-(pi)
    RETURN pi AS answer
"""
query_params = {"type": "MacPherson"}

# 执行查询并打印结果
answers = run_query(cypher_query, query_params)
print(f"问题: {question}")
for record in answers:
    print(f"- {record['answer']}")
