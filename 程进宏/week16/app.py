from config import Config
from zai import ZhipuAiClient
from py2neo import Graph
import json
import re
import pandas
import itertools

'''
使用LLM的方式进行知识图谱的使用
1. 提前构建三元组数据存入图数据库
2. 设计大模型提示词，让它能够按照数据格式识别和输出
3. 用户问题输入大模型拿到三元组和实体关系等（json数据）[实体entity、关系relation、属性attribute、标签label]
    3.1. 拿到如下格式的数据
LSTM的训练方式是什么？
{
  "entity": [
    {
      "id": 1,
      "name": "LSTM"
    }
  ],
  "relation": [
        {
          "ref": "属于",
          "subject_id": 1,
          "object_id": None
        }
],
  "attribute": [
    {
      "entity_id": 1,
      "key": "训练方式",
      "value": ""
    }
  ],
  "label": ["预训练模型"]
}
4. 解析json数据替换模板中的插槽（多实体的话进行排列组合在替换插槽），将替换后的问题模板与原问题进行相似度计算，排序拿到cypher依次查询图数据库
5. 将图数据库答案与问题模板插槽替换回答用户
'''


class ChatClient:
    def __init__(self, config):
        self.prompt = Config["prompt"]
        self.client = ZhipuAiClient(api_key=config["ZhiApiKey"])  # 初始化客户端

    def chat(self, u_content, s_content=Config["prompt"]):
        # 创建聊天完成请求
        response = self.client.chat.completions.create(
            model="glm-4.5",
            messages=[
                {
                    "role": "system",
                    "content": s_content
                },
                {
                    "role": "user",
                    "content": u_content
                }
            ],
            temperature=0.6
        )
        return response.choices[0].message.content


class GraphQA:
    def __init__(self):
        self.load_question_templet(Config["templet_path"])
        self.graph = Graph(Config["graph_profile"], auth=(Config["user"], Config["pw"]))
        self.client = ChatClient(Config)
        print("知识图谱问答系统加载完毕！\n===============")
    #加载模板
    # def load(self, schema_path, templet_path):
    #     # self.load_kg_schema(schema_path)
    #     self.load_question_templet(templet_path)
    #     return
    #
    # 加载模板信息（return question_templet列表）
    def load_question_templet(self, templet_path):
        dataframe = pandas.read_excel(templet_path)
        self.question_templet = []
        for index in range(len(dataframe)):
            question = dataframe["question"][index]
            cypher = dataframe["cypher"][index]
            cypher_check = dataframe["check"][index]
            answer = dataframe["answer"][index]
            self.question_templet.append([question, cypher, json.loads(cypher_check), answer])
        return

    # 处理json数据返回字典列表数据
    def handle_rsp_json(self, rsp):
        # 去除多余的markdown字符，得到纯净json字符串
        cleaned_rsp = re.sub(r'^```json\s*|\s*```$', '', rsp.strip(), flags=re.MULTILINE)
        # print(cleaned_rsp)
        # 解析json字符串, 得到实体entity、关系relation、属性attribute、标签label
        js = json.loads(cleaned_rsp)
        # 转换成实体关系属性标签列表不包含其他信息
        self.entitys = [i["name"] for i in js["entity"]]
        self.relations = [i["ref"] for i in js["relation"]]
        self.labels = js["label"]
        self.attributes = [i["key"] for i in js["attribute"]]
        return {"%ENT%": self.entitys,
                "%REL%": self.relations,
                "%LAB%": self.labels,
                "%ATT%": self.attributes}

    # 验证从文本种提取到的信息是否足够填充模板，如果不足够就跳过，节省运算速度
    # 如模板：  %ENT%和%ENT%是什么关系？  这句话需要两个实体才能填充，如果问题中只有一个，该模板无法匹配
    def check_cypher_info_valid(self, info, cypher_check):
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    # 根据提取到的实体，关系等信息，将模板展开成所有可能的待匹配的问题文本（经过实体排列组合替换模板后）
    def expand_question_and_cypher(self, info):
        templet_cypher_pair = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                templet_cypher_pair += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return templet_cypher_pair

    # 将提取到的值分配到键上
    # 输入参数:
    # value_combination = (("周杰伦", "方文山"), ("导演",))
    # cypher_check = {"%ENT%": 2, "%REL%": 1}
    # 执行过程:
    # index=0, key="%ENT%", required_count=2
    # 进入else分支:
    #   i=0: key_num = "%ENT" + "0" + "%" = "%ENT0%", res["%ENT0%"] = "周杰伦"
    #   i=1: key_num = "%ENT" + "1" + "%" = "%ENT1%", res["%ENT1%"] = "方文山"
    # index=1, key="%REL%", required_count=1
    # 进入if分支:
    #   res["%REL%"] = "导演"
    # 返回结果:
    # res = {"%ENT0%": "周杰伦", "%ENT1%": "方文山", "%REL%": "导演"}
    def decode_value_combination(self, value_combination, cypher_check):
        res = {}
        for index, (key, required_count) in enumerate(cypher_check.items()):
            if required_count == 1:
                res[key] = value_combination[index][0]
            else:
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    res[key_num] = value_combination[index][i]
        return res

    # 对于找到了超过模板中需求的实体数量的情况，需要进行 “排列组合”
    # info:{"%ENT%":["周杰伦", "方文山"], “%REL%”:[“作曲”]}
    def get_combinations(self, cypher_check, info):
        slot_values = []
        for key, required_count in cypher_check.items():
            slot_values.append(itertools.combinations(info[key], required_count))
            # 假设:
            # cypher_check = {"%ENT%": 2, "%REL%": 1}
            # info = {"%ENT%": ["周杰伦", "方文山", "不能说的秘密"], "%REL%": ["导演", "主演"]}
            # 执行过程:
            # 1. key="%ENT%", required_count=2
            #    itertools.combinations(["周杰伦", "方文山", "不能说的秘密"], 2)
            #    结果: [("周杰伦", "方文山"), ("周杰伦", "不能说的秘密"), ("方文山", "不能说的秘密")]
            #
            # 2. key="%REL%", required_count=1
            #    itertools.combinations(["导演", "主演"], 1)
            #    结果: [("导演",), ("主演",)]
            #
            # slot_values = [
            #   [("周杰伦", "方文山"), ("周杰伦", "不能说的秘密"), ("方文山", "不能说的秘密")],
            #   [("导演",), ("主演",)]
            # ]
        # 对各槽位的组合进行笛卡尔积运算，生成所有可能的组合：
        value_combinations = itertools.product(*slot_values)
        # 继续上面的示例:
        # itertools.product(*slot_values) 等价于 itertools.product(slot_values[0], slot_values[1])
        # 结果:
        # value_combinations = [
        #     (("周杰伦", "方文山"), ("导演",)),
        #     (("周杰伦", "方文山"), ("主演",)),
        #     (("周杰伦", "不能说的秘密"), ("导演",)),
        #     (("周杰伦", "不能说的秘密"), ("主演",)),
        #     (("方文山", "不能说的秘密"), ("导演",)),
        #     (("方文山", "不能说的秘密"), ("主演",))
        # ]
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
        return combinations

    # 将带有token的模板替换成真实词
    # string:%ENT1%和%ENT2%是%REL%关系吗
    # combination: {"%ENT1%":"word1", "%ENT2%":"word2", "%REL%":"word"}
    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            # 处理不同类型的值
            if value is None:
                replacement = ""
            elif isinstance(value, (int, float)):
                replacement = str(value)
            else:
                replacement = str(value)
            string = string.replace(key, replacement)
        return string

    # 对于单条模板，根据抽取到的实体属性信息扩展，形成一个列表
    # info:{"%ENT%":["周杰伦", "方文山"], “%REL%”:[“作曲”]}
    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        combinations = self.get_combinations(cypher_check, info)  # 如果多实体排列组合后的字典列表
        templet_cpyher_pair = []
        for combination in combinations:
            # 将模板中的问题插槽替换成问题中的实体、属性、关系、标签
            replaced_templet = self.replace_token_in_string(templet, combination)
            # 将模板中的sql语法插槽替换成问题中的实体、属性、关系、标签
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            # 将模板中的答案插槽替换成问题中的实体、属性、关系、标签
            replaced_answer = self.replace_token_in_string(answer, combination)
            templet_cpyher_pair.append([replaced_templet, replaced_cypher, replaced_answer])
        return templet_cpyher_pair  # 返回插槽替换后的模板

    # 距离函数，文本匹配的所有方法都可以使用
    def sentence_similarity_function(self, string1, string2):
        # print("计算  %s %s"%(string1, string2))
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance

    # 通过问题匹配的方式确定匹配的所有cypher语句
    def cypher_match(self, sentence, info):
        templet_cypher_pair = self.expand_question_and_cypher(info)  # 模板展开，获取所有可能的模板
        # print(templet_cypher_pair)
        result = []
        for templet, cypher, answer in templet_cypher_pair:
            score = self.sentence_similarity_function(sentence, templet)
            # print(sentence, templet, score)
            result.append([templet, cypher, score, answer])
        result = sorted(result, reverse=True, key=lambda x: x[2])
        return result
        # 排列组合
        # 替换模板
        # 相似度计算
        # sql查询
        # 答案替换

    # 解析结果
    def parse_result(self, graph_search_result, answer, info):
        graph_search_result = graph_search_result[0]
        # 关系查找返回的结果形式较为特殊，单独处理
        if "REL" in graph_search_result:
            graph_search_result["REL"] = list(graph_search_result["REL"].types())[0]
        answer = self.replace_token_in_string(answer, graph_search_result)
        return answer

    def query(self, q):
        print("============")
        print(q)
        rsp = self.client.chat(q)
        # 对提取的json数据进行预处理
        info = self.handle_rsp_json(rsp)
        print("info:", info)
        templet_cypher_score = self.cypher_match(q, info)  # cypher匹配
        for templet, cypher, score, answer in templet_cypher_score:
            graph_search_result = self.graph.run(cypher).data()
            # 最高分命中的模板不一定在图上能找到答案, 当不能找到答案时，运行下一个搜索语句, 找到答案时停止查找后面的模板
            if graph_search_result:
                answer = self.parse_result(graph_search_result, answer, info)
                return answer
        return None

    def search_graph(self, cypher):
        graph_search_result = self.graph.run(cypher).data()
        if graph_search_result:
            return graph_search_result[0]
        return None


def test():
    g = GraphQA()
    a = g.query("Match ({NAME:'BERT'})-[:属于]->(n) return n.NAME")
    print(str(a) + "sjadks")


if __name__ == "__main__":
    # glm = ChatClient(Config)

    qa = GraphQA()
    # query = "GPT_3的开发机构是什么"
    query = "Transformer模型的论文标题是什么？"
    answer = qa.query(query)
    print("用户问题：" + str(query))
    print("回答：" + str(answer))
    # test()
