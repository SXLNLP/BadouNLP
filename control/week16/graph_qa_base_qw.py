import json
import os
from openai import OpenAI
from py2neo import Graph


def askk_question(question):
    # apikey = os.getenv("DASHSCOPE_API_KEY")
    apikey = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxx"
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=apikey,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{question}"},
        ],
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
    )
    answer =  completion.choices[0].message.content
    return answer


prompt = '''
请根据知识图谱的信息:
       Dynasty（朝代）节点，具有name（朝代名字），start_year（朝代开始日），end_year（朝代结束日） ；
       Poet（诗人） 节点，具有name（诗人名字），courtesy_name（字），birth_place（出生地），intro（诗人介绍），birth_year（出生年份），death_year（死亡年份）；
       Poem （诗词）节点，具有title（诗名），content(诗句），writing_style（写作意境）；
       诗人和朝代建立了关系，比如：Poet {name: "李白"})-[:BELONGS_TO]Dynasty {name: "唐"}
       诗人和诗词建立关系，比如：Poet {name: "李白"}-[:WRITES]-Poem {title: "望庐山瀑布"}
分析最后的客户的问题，生成对应的符合neo4j语法的语句
并组织成严格的 JSON 格式。
JSON Schema 要求：
- query (string): 符合neo4j语法的查询语句
- result (string): 从返回的查询结果中提取结果，结果的key
只输出 JSON格式的数据，不要有json标识等任何额外内容。不要有json标识等任何额外内容。
如：
{
"query":"",
"result":""
}
客户问题：
'''
def askGraph(query):
    #调用neo4j,执行命令查询

    graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "12345678"))

    graph_search_result = graph.run(query).data()
    return graph_search_result

def parsejson(f):
    # 先读取文件内容，再用 loads() 解析
    result = json.loads(f)
    return result
def askQuestion(question):
    print(f"问题：{question}")
    findQuery = askk_question(prompt + question)
    queryPair = parsejson(findQuery)
    print(f"query：{queryPair["query"]}")
    result = askGraph(queryPair["query"])
    print("答案是：")
    for item in result:
        print(item[queryPair["result"]])

if __name__ == "__main__":

    question1 = "李白写了哪些诗词？"
    askQuestion(question1)
    question2 = "静夜思的诗句是什么？"
    askQuestion(question2)
    question3 = " 哪个朝代的开始日是 617 到 619 之间?"
    askQuestion(question3)