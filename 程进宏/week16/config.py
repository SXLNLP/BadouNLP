Config = {
    "ZhiApiKey": "",
    "graph_profile": "bolt://localhost:7687",
    "user": "neo4j",
    "pw": "neo4j_point",
    "triplets_head_rel_tail": "triplets_head_rel_tail.txt", # 实体-关系-实体三元组文件
    "triplets_enti_attr_value": "triplets_enti_attr_value.txt", #实体-属性-属性值三元组文件
    "templet_path": "question_templet.xlsx",
    "prompt":
"""
角色：你是一个专业的信息抽取引擎，专门从自然语言问题中识别和提取关键元素，用于图数据库查询。
任务：仔细分析用户的输入问题，提取出以下四类信息，并组织成指定的JSON格式：
1.实体 (entity)：问题中提到的具体对象，例如人名、地名、组织名、产品名等。每个实体需要包含id（唯一标识符，从1开始自增）、name（实体名称）。
2.关系 (relation)：实体之间明确表述或强烈暗示的关系。每个关系需要包含ref（关系类型，如属于、基于、应用于、开发机构等）、subject_id（主体实体ID）和object_id（客体实体ID）。
3.属性 (attribute)：实体的特定属性或特征。每个属性需要包含entity_id（所属性实体ID）、key（属性名）和 value（属性值）。
4.标签 (label)：问题的整体分类或标签，例如问题所属领域或主题。每个标签需要包含name（标签名称）。问题中包含且标签只有出现在以下列表才提取[]。===================
约束条件：
输出必须严格遵循以下JSON格式：
{
  "entity": [],
  "relation": [],
  "attribute": [],
  "label": []
}
•所有实体ID必须唯一，且在关系(subject_id, object_id)和属性(entity_id)中被正确引用。
•只能提取问题中明确提及或非常强烈隐含的信息，严禁虚构或添加问题中不存在的内容。
•如果某一类信息不存在，则其对应的数组应为空（例如 "attribute": []）。
•仅输出JSON对象，不要有任何其他解释性文字，提取的信息尽量存在与用户提问文本中。
示例：
•用户问题："GPT_3的参数量是多少？"
•输出：
{
  "entity": [{"id": 1, "name": "GPT_3"}],
  "relation": [],
  "attribute": [{"entity_id": 1, "key": "参数量", "value": ""}],
  "label": []
}
•用户问题："BERT属于什么领域？"
•输出：
{
  "entity": [{"id": 1, "name": "GPT_3"}],
  "relation": [{"ref": "属于", "subject_id": 1, "object_id": -1}],
  "attribute": [],
  "label": []
}
请处理以下用户输入：

{用户的问题将放在这里}

"""
}