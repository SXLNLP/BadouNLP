import re
import json
import pandas as pd
from py2neo import Graph
from collections import defaultdict
import itertools


class GraphQA:
    def __init__(self, uri="http://localhost:7474", user="neo4j", password="881201",
                 schema_path="kg_schema.json", templet_path="question_templet_fixed.xlsx"):
        """
        初始化问答系统
        """
        self.graph = Graph(uri, auth=(user, password))
        self.entity_set = set()
        self.relation_set = set()
        self.attribute_set = set()
        self.label_set = set()
        self.question_templates = []

        # 加载 schema 和模板
        self.load_schema(schema_path)
        self.load_question_templet(templet_path)
        print("✅ 知识图谱问答系统加载完毕！")

    def load_schema(self, schema_path):
        """加载 kg_schema.json"""
        with open(schema_path, "r", encoding="utf8") as f:
            schema = json.load(f)
        self.entity_set = set(schema.get("entitys", []))
        self.relation_set = set(schema.get("relations", []))
        self.attribute_set = set(schema.get("attributes", []))
        self.label_set = set(schema.get("labels", []))

    def load_question_templet(self, templet_path):
        """从 Excel 加载问答模板"""
        try:
            df = pd.read_excel(templet_path)
            self.question_templates = []
            for _, row in df.iterrows():
                question = str(row["question"]).strip()
                cypher = str(row["cypher"]).strip()
                # 解析 check 字段
                try:
                    cypher_check = json.loads(str(row["check"]))
                except:
                    cypher_check = {"%ENT%": 1}  # 默认
                answer = str(row["answer"]).strip()
                self.question_templates.append([question, cypher, cypher_check, answer])
            print(f"✅ 成功加载 {len(self.question_templates)} 个模板")
        except Exception as e:
            print(f"❌ 加载模板失败：{e}")
            self.question_templates = []

    def extract_mentions(self, sentence):
        """提取句子中提到的实体、关系、属性"""
        entities = list(set(re.findall("|".join(re.escape(e) for e in self.entity_set), sentence)))
        relations = list(set(re.findall("|".join(re.escape(r) for r in self.relation_set), sentence)))
        attributes = list(set(re.findall("|".join(re.escape(a) for a in self.attribute_set), sentence)))
        return {
            "%ENT%": entities,
            "%REL%": relations,
            "%ATT%": attributes
        }

    def is_slot_valid(self, info, template_slots):
        """检查信息是否满足模板需求"""
        for slot, required_count in template_slots.items():
            if len(info.get(slot, [])) < required_count:
                return False
        return True

    def generate_combinations(self, info, template_slots):
        """生成所有可能的占位符组合"""
        candidates = []
        for slot, required_count in template_slots.items():
            values = info.get(slot, [])
            if required_count == 1:
                candidates.append([(v,) for v in values])
            else:
                candidates.append(itertools.combinations(values, required_count))
        for combo in itertools.product(*candidates):
            mapping = {}
            for slot, value_tuple in zip(template_slots.keys(), combo):
                if len(value_tuple) == 1:
                    mapping[slot] = value_tuple[0]
                else:
                    for i, v in enumerate(value_tuple):
                        mapping[f"{slot[:-1]}{i+1}%"] = v
            yield mapping

    def fill_template(self, template, mapping):
        """填充模板"""
        filled = template
        for k, v in mapping.items():
            filled = filled.replace(k, v)
        return filled

    def execute_cypher(self, cypher):
        """执行 Cypher 并返回结果字符串"""
        try:
            result = self.graph.run(cypher).data()
            if not result or not result[0]:
                return None
            # 尝试提取多种可能的返回字段
            record = result[0]
            # 常见字段：result, n.NAME, children, movies, songs...
            value = None
            for key in ['result'] + list(record.keys()):
                if key != 'result' and isinstance(record[key], (list, str)):
                    value = record[key]
                    break
                elif key == 'result':
                    value = record[key]
                    break
            if value is None:
                return None
            if isinstance(value, list):
                return "、".join(str(x) for x in value)
            return str(value) if value else None
        except Exception as e:
            print("❌ Cypher 执行错误：", e)
            return None

    def query(self, question):
        print(f"\n🔍 问题：{question}")
        info = self.extract_mentions(question)

        best_match = None
        best_score = 0.6  # 阈值

        for template in self.question_templates:
            template_question, cypher, cypher_check, answer = template

            if not self.is_slot_valid(info, cypher_check):
                continue

            for mapping in self.generate_combinations(info, cypher_check):
                filled_question = self.fill_template(template_question, mapping)
                filled_cypher = self.fill_template(cypher, mapping)
                filled_answer = self.fill_template(answer, mapping)

                score = self.similarity(question, filled_question)
                if score > best_score:
                    best_score = score
                    best_match = (filled_question, filled_cypher, filled_answer)

        if best_match:
            filled_question, filled_cypher, filled_answer = best_match
            print(f"🎯 匹配模板：{filled_question}")
            print(f"🧩 Cypher：{filled_cypher}")
            answer_value = self.execute_cypher(filled_cypher)
            if answer_value:
                # ✅ 修复：尝试多种可能的 result 格式
                possible_patterns = [
                    "`result`",  # 正确格式
                    "'result'",  # 常见错误格式
                    '"result"',  # 常见错误格式
                    "result",    # 没有引号
                    "` result `", # 有空格
                    "' result '", # 有空格
                ]
                
                final_answer = filled_answer
                for pattern in possible_patterns:
                    if pattern in final_answer:
                        final_answer = final_answer.replace(pattern, answer_value)
                        break
                
                print(f"✅ 回答：{final_answer}")
                return final_answer

        print("❌ 未找到匹配答案")
        return "抱歉，我暂时无法回答这个问题。"

    def similarity(self, s1, s2):
        """简单 Jaccard 相似度"""
        set1, set2 = set(s1), set(s2)
        return len(set1 & set2) / (len(set1 | set2) + 1e-8)


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    qa = GraphQA()

    # 测试问题
    questions = [
        "发如雪的谱曲是谁",
        "谢霆锋的血型是什么",
        "谢霆锋的身高是多少",
        "谢霆锋的配偶是谁",
        "谢霆锋的父亲是谁",
        "谢霆锋的孩子有哪些",
        "谢霆锋的职业是什么",
        "谢霆锋唱过哪些歌",
        "谢霆锋出过哪些专辑",
        "谢霆锋参演过哪些电影",
        "周杰伦的星座是什么",
        "王菲和谢霆锋是什么关系",
    ]

    for q in questions:
        qa.query(q)

# import pandas as pd
# import re
# import json
# import os

# def convert_question_templates(input_path, output_path):
#     """
#     修复版：知识图谱问答模板转换脚本
#     1. 正确处理变量名和标签
#     2. 修复所有语法问题
#     3. 确保变量名一致性
#     """
#     print(f"🔍 正在处理模板文件: {input_path}")
    
#     try:
#         df = pd.read_excel(input_path)
#         print(f"✅ 成功加载 {len(df)} 个模板")
#     except Exception as e:
#         print(f"❌ 无法读取 Excel 文件: {e}")
#         return False
    
#     required_columns = ['question', 'cypher', 'answer', 'check']
#     missing_cols = [col for col in required_columns if col not in df.columns]
#     if missing_cols:
#         print(f"❌ 错误：缺少必要列 {missing_cols}")
#         return False
    
#     fixed_templates = []
    
#     for idx, row in df.iterrows():
#         try:
#             question = str(row['question']).strip()
#             cypher = str(row['cypher']).strip()
#             answer = str(row['answer']).strip()
#             check = str(row['check']).strip()
            
#             # 1. ✅ 修复变量名和标签问题（关键修复！）
#             # 正确保留变量名，只移除标签
#             # 示例: (p:Person { → (p {
#             cypher = re.sub(r'\(\s*(\w+):\w+\s*\{', r'(\1 {', cypher)
            
#             # 2. ✅ 修复方括号闭合问题
#             cypher = re.sub(r'\[:([^\]]+)-', r'[:\1]', cypher)
#             cypher = re.sub(r'\[:([^\]]+)\s*([^\]])', r'[:\1]\2', cypher)
            
#             # 3. ✅ 移除非法字符
#             cypher = cypher.replace(']', '').replace('->(m)', '').replace('->(n)', '')
            
#             # 4. ✅ 修复方向问题（保留原始方向）
#             # 将错误的 ]->(m) 替换为正确的方向
#             cypher = re.sub(r'\]\s*->\s*\(\w+\)', '', cypher)
#             cypher = re.sub(r'<-\s*\(\s*\{', r'<- (m {', cypher)
#             cypher = re.sub(r'->\s*\(\s*\{', r'-> (m {', cypher)
            
#             # 5. ✅ 确保变量名存在（关键！）
#             # 如果没有变量名，添加默认变量名 m
#             cypher = re.sub(r'\(\s*\{', r'(m {', cypher)
            
#             # 6. ✅ 统一返回字段为 'result'
#             # 处理单值返回
#             cypher = re.sub(
#                 r'RETURN\s+(\w+\.\w+)(?:\s+as\s+\w+)?', 
#                 r'RETURN \1 AS result', 
#                 cypher, 
#                 flags=re.IGNORECASE
#             )
#             # 处理列表返回
#             cypher = re.sub(
#                 r'RETURN\s+collect\((\w+\.\w+)\)(?:\s+as\s+\w+)?', 
#                 r'RETURN collect(\1) AS result', 
#                 cypher, 
#                 flags=re.IGNORECASE
#             )
#             # 处理 WHERE 子句
#             cypher = re.sub(r'(WHERE\s+[^)]+)\s+RETURN', r'\1 RETURN', cypher)
            
#             # 7. ✅ 修复答案模板
#             if '`result`' not in answer:
#                 if any(kw in answer for kw in ['职业', '血型', '星座', '身高', '国籍', '籍贯', '婚姻状况']):
#                     answer = f'%ENT%的{answer}是`result`'
#                 elif any(kw in answer for kw in ['孩子', '好友', '合作', '作品', '专辑', '奖项', '企业', '品牌']):
#                     answer = f'%ENT%的{answer}有：`result`'
#                 elif answer in ['配偶', '妻子', '丈夫', '父亲', '母亲', '儿子', '女儿']:
#                     answer = f'%ENT%的{answer}是`result`'
#                 else:
#                     answer = f'%ENT%{answer}是`result`'
            
#             # 8. ✅ 修复 check 字段
#             try:
#                 check_dict = json.loads(check)
#                 if not isinstance(check_dict, dict):
#                     check_dict = {"%ENT%": 1}
#             except:
#                 check_dict = {"%ENT%": 1}
            
#             # 9. ✅ 验证修复后的 Cypher
#             if not validate_cypher_syntax(cypher):
#                 print(f"⚠️ 跳过无效模板 (行 {idx+2}): {question}")
#                 continue
            
#             fixed_templates.append({
#                 'question': question,
#                 'cypher': cypher,
#                 'answer': answer,
#                 'check': json.dumps(check_dict, ensure_ascii=False)
#             })
            
#         except Exception as e:
#             print(f"❌ 处理模板时出错 (行 {idx+2}): {e}")
#             continue
    
#     # 保存修复后的模板
#     try:
#         os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
#         fixed_df = pd.DataFrame(fixed_templates)
#         fixed_df.to_excel(output_path, index=False)
        
#         print(f"✅ 成功生成修复后的模板文件: {output_path}")
#         print(f"✅ 共处理 {len(df)} 个模板，成功修复 {len(fixed_df)} 个")
        
#         # 打印验证示例
#         print("\n🔍 验证示例:")
#         for i in range(min(3, len(fixed_df))):
#             print(f"  问题: {fixed_df.iloc[i]['question']}")
#             print(f"  修复后Cypher: {fixed_df.iloc[i]['cypher']}")
#             print(f"  修复后答案: {fixed_df.iloc[i]['answer']}\n")
        
#         return True
    
#     except Exception as e:
#         print(f"❌ 保存文件失败: {e}")
#         return False

# def validate_cypher_syntax(cypher):
#     """验证 Cypher 语法"""
#     # 检查基本结构
#     if not (cypher.startswith('MATCH') and 'RETURN' in cypher):
#         return False
    
#     # 检查方括号闭合
#     if cypher.count('[') != cypher.count(']'):
#         return False
    
#     # 检查变量名一致性
#     # 提取 MATCH 中的变量名
#     match_vars = re.findall(r'\(\s*(\w+)\s*\{', cypher)
#     if not match_vars:
#         return False
    
#     # 检查 RETURN 中是否使用了这些变量名
#     return any(var in cypher for var in match_vars)

# # ========================
# # 使用示例
# # ========================
# if __name__ == "__main__":
#     input_file = "question_templet.xlsx"
#     output_file = "question_templet_fixed.xlsx"
    
#     print("=" * 50)
#     print("🚀 开始转换知识图谱问答模板")
#     print("=" * 50)
    
#     success = convert_question_templates(input_file, output_file)
    
#     if success:
#         print("\n" + "=" * 50)
#         print("✅ 转换成功！修复了以下关键问题：")
#         print("   - 变量名与标签处理错误（如 (Person { → (p {）")
#         print("   - 变量名不一致（MATCH 中的变量 vs RETURN 中的变量）")
#         print("   - 方括号未闭合问题")
#         print("   - 非法字符 ] 问题")
#         print("GraphQA(templet_path='question_templet_fixed.xlsx')")
#         print("=" * 50)
#     else:
#         print("\n" + "=" * 50)
#         print("❌ 转换失败，请检查错误信息")
#         print("=" * 50)

# import pandas as pd
# import re
# import json

# def fix_question_templates():
#     """一键修复模板文件"""
#     try:
#         # 1. 读取原始模板
#         df = pd.read_excel("question_templet_fixed.xlsx")
#         print(f"✅ 成功加载 {len(df)} 个模板")
        
#         # 2. 创建修复后的模板
#         fixed_data = []
        
#         # 3. 修复每个模板
#         for _, row in df.iterrows():
#             question = str(row["question"]).strip()
#             cypher = str(row["cypher"]).strip()
#             answer = str(row["answer"]).strip()
#             check = str(row["check"]).strip()
            
#             # 修复答案模板：确保 result 用反引号包裹
#             if '`result`' not in answer:
#                 # 尝试修复各种错误格式
#                 answer = re.sub(r"['\"]?result['\"]?", "`result`", answer)
#                 # 特殊处理
#                 if '配偶' in question:
#                     answer = '%ENT%的配偶是`result`'
#                 elif '孩子' in question or '子女' in question:
#                     answer = '%ENT%的孩子有：`result`'
#                 elif '专辑' in question:
#                     answer = '%ENT%的专辑有：`result`'
#                 elif '电影' in question:
#                     answer = '%ENT%参演的电影有：`result`'
#                 elif '关系' in question and '%ENT1%' in question:
#                     answer = '%ENT1%和%ENT2%是`result`关系'
#                 elif '谱曲' in question:
#                     answer = '%ENT%的谱曲是`result`'
            
#             # 修复关系名称（根据常见问题）
#             if '参演作品' in cypher and '参演电影' not in answer:
#                 cypher = cypher.replace('[:参演作品]', '[:参演电影]')
            
#             # 修复双实体查询模板
#             if '%ENT1%' in question and '%ENT2%' in question and '关系' in question:
#                 cypher = 'MATCH (e1 {NAME:"%ENT1%"})-[r]-(e2 {NAME:"%ENT2%"}) RETURN type(r) AS result'
#                 check = '{"%ENT1%":1, "%ENT2%":1}'
            
#             # 修复 check 字段
#             try:
#                 check_dict = json.loads(check)
#             except:
#                 check_dict = {"%ENT%": 1}
            
#             fixed_data.append([
#                 question,
#                 cypher,
#                 answer,
#                 json.dumps(check_dict, ensure_ascii=False)
#             ])
        
#         # 4. 保存修复后的模板
#         fixed_df = pd.DataFrame(fixed_data, 
#                                columns=["question", "cypher", "answer", "check"])
#         output_file = "question_templet_fixed_correct.xlsx"
#         fixed_df.to_excel(output_file, index=False)
        
#         print(f"✅ 模板修复成功！已保存到 {output_file}")
#         print("\n修复示例：")
#         print(f"  问题: {fixed_data[0][0]}")
#         print(f"  修复后答案: {fixed_data[0][2]}")
#         print(f"  应该显示: 谢霆锋的配偶是王菲 (而不是 'result')")
        
#         # 5. 打印验证指南
#         print("\n🔍 验证步骤：")
#         print("1. 确保答案模板中 result 被反引号包裹：`result`")
#         print("2. 检查关系名称是否与图谱匹配（在 Neo4j 中运行验证查询）")
#         print("3. 添加缺失的模板（如双实体关系查询）")
#         print("4. 使用增强版 GraphQA.query 方法")
        
#         return True
    
#     except Exception as e:
#         print(f"❌ 修复失败：{e}")
#         return False

# if __name__ == "__main__":
#     print("=" * 50)
#     print("🚀 一键修复知识图谱问答模板")
#     print("=" * 50)
    
#     success = fix_question_templates()
    
#     if success:
#         print("\n" + "=" * 50)
#         print("✅ 修复完成！请执行以下步骤：")
#         print("1. 将 GraphQA 指向修复后的模板：")
#         print("   GraphQA(templet_path='question_templet_fixed_correct.xlsx')")
#         print("2. 使用增强版 query 方法（已处理多种 result 格式）")
#         print("3. 在 Neo4j 中验证关系名称是否匹配")
#         print("=" * 50)
#     else:
#         print("\n" + "=" * 50)
#         print("❌ 修复失败，请检查错误信息")
#         print("=" * 50)