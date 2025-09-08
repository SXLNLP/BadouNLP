import pandas as pd
import json

# 创建问题模板数据
questions_data = [
    {
        "question": "%ENT%的作用是什么",
        "cypher": "MATCH (n) WHERE n.NAME='%ENT%' RETURN n.作用 as answer",
        "check": '{"％ENT%": 1}',
        "answer": "%ENT%的作用是%answer%"
    },
    {
        "question": "%ENT%函数应用于什么场景",
        "cypher": "MATCH (n)-[:应用于]->(m) WHERE n.NAME='%ENT%' RETURN m.NAME as answer",
        "check": '{"％ENT%": 1}',
        "answer": "%ENT%函数应用于%answer%场景"
    },
    {
        "question": "%ENT%的使用频率如何",
        "cypher": "MATCH (n) WHERE n.NAME='%ENT%' RETURN n.使用频率 as answer",
        "check": '{"％ENT%": 1}',
        "answer": "%ENT%的使用频率是%answer%"
    },
    {
        "question": "%ENT%的参数是什么",
        "cypher": "MATCH (n) WHERE n.NAME='%ENT%' RETURN n.参数 as answer",
        "check": '{"％ENT%": 1}',
        "answer": "%ENT%的参数是%answer%"
    },
    {
        "question": "%ENT%的返回值是什么",
        "cypher": "MATCH (n) WHERE n.NAME='%ENT%' RETURN n.返回值 as answer",
        "check": '{"％ENT%": 1}',
        "answer": "%ENT%的返回值是%answer%"
    },
    {
        "question": "哪些函数属于%ENT%",
        "cypher": "MATCH (m)-[:包含]->(n) WHERE m.NAME='%ENT%' RETURN n.NAME as answer",
        "check": '{"％ENT%": 1}',
        "answer": "%answer%属于%ENT%"
    },
    {
        "question": "%ENT%和%ENT1%是什么关系",
        "cypher": "MATCH (n)-[r]-(m) WHERE n.NAME='%ENT%' AND m.NAME='%ENT1%' RETURN type(r) as answer",
        "check": '{"％ENT%": 1, "％ENT1%": 1}',
        "answer": "%ENT%和%ENT1%的关系是%answer%"
    }
]

# 创建DataFrame
df = pd.DataFrame(questions_data)
df.to_excel("question_templet.xlsx", index=False)

print("问题模板文件创建完成！")