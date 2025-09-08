from py2neo import Graph, Node, Relationship, NodeMatcher
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 连接到Neo4j数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))  # 替换为你的连接信息

# 清除现有数据
graph.delete_all()

# 创建示例电影知识图谱
def create_sample_graph():
    # 创建电影节点
    matrix = Node("Movie", title="The Matrix", released=1999, tagline="Welcome to the Real World")
    matrix_reloaded = Node("Movie", title="The Matrix Reloaded", released=2003)
    matrix_revolutions = Node("Movie", title="The Matrix Revolutions", released=2003)
    
    # 创建人物节点
    keanu = Node("Person", name="Keanu Reeves", born=1964)
    laurence = Node("Person", name="Laurence Fishburne", born=1961)
    carrie = Node("Person", name="Carrie-Anne Moss", born=1967)
    hugo = Node("Person", name="Hugo Weaving", born=1960)
    lana = Node("Person", name="Lana Wachowski", born=1965)
    lily = Node("Person", name="Lilly Wachowski", born=1967)
    joel = Node("Person", name="Joel Silver", born=1952)
    
    # 创建关系
    acts_in = "ACTED_IN"
    directed = "DIRECTED"
    produced = "PRODUCED"
    
    graph.create(Relationship(keanu, acts_in, matrix, roles=["Neo"]))
    graph.create(Relationship(keanu, acts_in, matrix_reloaded, roles=["Neo"]))
    graph.create(Relationship(keanu, acts_in, matrix_revolutions, roles=["Neo"]))
    graph.create(Relationship(laurence, acts_in, matrix, roles=["Morpheus"]))
    graph.create(Relationship(laurence, acts_in, matrix_reloaded, roles=["Morpheus"]))
    graph.create(Relationship(laurence, acts_in, matrix_revolutions, roles=["Morpheus"]))
    graph.create(Relationship(carrie, acts_in, matrix, roles=["Trinity"]))
    graph.create(Relationship(carrie, acts_in, matrix_reloaded, roles=["Trinity"]))
    graph.create(Relationship(carrie, acts_in, matrix_revolutions, roles=["Trinity"]))
    graph.create(Relationship(hugo, acts_in, matrix, roles=["Agent Smith"]))
    graph.create(Relationship(hugo, acts_in, matrix_reloaded, roles=["Agent Smith"]))
    graph.create(Relationship(hugo, acts_in, matrix_revolutions, roles=["Agent Smith"]))
    graph.create(Relationship(lana, directed, matrix))
    graph.create(Relationship(lily, directed, matrix))
    graph.create(Relationship(lana, directed, matrix_reloaded))
    graph.create(Relationship(lily, directed, matrix_reloaded))
    graph.create(Relationship(lana, directed, matrix_revolutions))
    graph.create(Relationship(lily, directed, matrix_revolutions))
    graph.create(Relationship(joel, produced, matrix))
    graph.create(Relationship(joel, produced, matrix_reloaded))
    graph.create(Relationship(joel, produced, matrix_revolutions))
    
    print("示例知识图谱创建完成！")

# 创建示例图谱
create_sample_graph()

# 自然语言问题处理函数
def process_question(question):
    # 简单的问题分类和Cypher查询生成
    question_lower = question.lower()
    
    # 匹配演员出演的电影
    if "电影" in question_lower and "演" in question_lower:
        match = re.search(r'(.+?)演了哪些电影', question)
        if match:
            actor = match.group(1).strip()
            return f"""
            MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
            WHERE p.name = '{actor}'
            RETURN m.title AS movie
            """
    
    # 匹配电影的演员
    elif "演员" in question_lower or "谁演" in question_lower:
        match = re.search(r'《?(.+?)》?的演员', question)
        if not match:
            match = re.search(r'谁演了《?(.+?)》?', question)
        if match:
            movie = match.group(1).strip()
            return f"""
            MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
            WHERE m.title = '{movie}'
            RETURN p.name AS actor, r.roles AS roles
            """
    
    # 匹配导演的电影
    elif "导演" in question_lower and "电影" in question_lower:
        match = re.search(r'(.+?)导演了哪些电影', question)
        if match:
            director = match.group(1).strip()
            return f"""
            MATCH (p:Person)-[:DIRECTED]->(m:Movie)
            WHERE p.name = '{director}'
            RETURN m.title AS movie
            """
    
    # 匹配电影的导演
    elif "导演" in question_lower:
        match = re.search(r'《?(.+?)》?的导演', question)
        if match:
            movie = match.group(1).strip()
            return f"""
            MATCH (p:Person)-[:DIRECTED]->(m:Movie)
            WHERE m.title = '{movie}'
            RETURN p.name AS director
            """
    
    # 默认查询 - 搜索相关信息
    tokens = word_tokenize(question)
    tagged = pos_tag(tokens)
    
    # 提取名词和专有名词
    entities = [word for word, pos in tagged if pos in ['NN', 'NNP', 'NNS']]
    
    if entities:
        entity = entities[0]
        return f"""
        MATCH (n)
        WHERE n.name CONTAINS '{entity}' OR n.title CONTAINS '{entity}'
        RETURN n
        LIMIT 10
        """
    
    return None

# 问答函数
def ask_question(question):
    print(f"问题: {question}")
    
    # 处理问题并生成Cypher查询
    cypher_query = process_question(question)
    
    if not cypher_query:
        print("抱歉，我不理解这个问题。")
        return
    
    print(f"生成的Cypher查询: {cypher_query}")
    
    try:
        # 执行查询
        result = graph.run(cypher_query).data()
        
        if not result:
            print("没有找到相关信息。")
            return
        
        # 显示结果
        print("答案:")
        for record in result:
            print(record)
            
    except Exception as e:
        print(f"查询出错: {e}")

# 测试问答系统
questions = [
    "Keanu Reeves演了哪些电影",
    "《The Matrix》的演员有哪些",
    "谁演了The Matrix Reloaded",
    "Lana Wachowski导演了哪些电影",
    "《The Matrix》的导演是谁",
    "Carrie-Anne Moss的信息"
]

for q in questions:
    ask_question(q)
    print("-" * 50)
