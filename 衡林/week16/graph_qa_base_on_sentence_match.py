import re
import json
import pandas as pd
import itertools
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from py2neo import Graph
from collections import defaultdict
from functools import lru_cache
import Levenshtein  # 需要安装: pip install python-Levenshtein

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphQA:
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化知识图谱问答系统

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._initialize_graph()
        self._load_schema_and_templates()
        logger.info("知识图谱问答系统加载完毕！\n===============")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            "neo4j_uri": "bolt://127.0.0.1:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "yzyzzz999",
            "neo4j_name": "neo4j",
            "schema_path": "kg_schema.json",
            "template_path": "question_templet.xlsx",
            "similarity_threshold": 0.3,
            "max_combinations": 100,
            "cache_size": 1000
        }

        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"加载配置文件失败: {e}, 使用默认配置")

        return default_config

    def _initialize_graph(self):
        """初始化图数据库连接"""
        try:
            self.graph = Graph(
                self.config["neo4j_uri"],
                user=self.config["neo4j_user"],
                password=self.config["neo4j_password"],
                name=self.config["neo4j_name"]
            )
            # 测试连接
            self.graph.run("RETURN 1 AS test")
            logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise

    def _load_schema_and_templates(self):
        """加载图谱模式和问题模板"""
        try:
            self.load_kg_schema(self.config["schema_path"])
            self.load_question_templet(self.config["template_path"])
        except Exception as e:
            logger.error(f"加载模式或模板失败: {e}")
            raise

    def load_kg_schema(self, path: str):
        """加载图谱模式信息"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                schema = json.load(f)

            self.relation_set = set(schema.get("relations", []))
            self.entity_set = set(schema.get("entitys", []))
            self.label_set = set(schema.get("labels", []))
            self.attribute_set = set(schema.get("attributes", []))

            logger.info(f"加载模式成功: {len(self.entity_set)}实体, "
                        f"{len(self.relation_set)}关系, "
                        f"{len(self.label_set)}标签, "
                        f"{len(self.attribute_set)}属性")
        except Exception as e:
            logger.error(f"加载模式文件失败: {e}")
            raise

    def load_question_templet(self, template_path: str):
        """加载问题模板"""
        try:
            dataframe = pd.read_excel(template_path)
            self.question_templet = []

            for index, row in dataframe.iterrows():
                try:
                    question = row["question"]
                    cypher = row["cypher"]
                    cypher_check = json.loads(row["check"])
                    answer = row["answer"]
                    self.question_templet.append([question, cypher, cypher_check, answer])
                except Exception as e:
                    logger.warning(f"处理模板第{index + 1}行时出错: {e}")

            logger.info(f"成功加载 {len(self.question_templet)} 个问题模板")
        except Exception as e:
            logger.error(f"加载模板文件失败: {e}")
            raise

    @lru_cache(maxsize=1000)
    def get_mention_entities(self, sentence: str) -> List[str]:
        """获取问题中提到的实体（带缓存）"""
        return self._find_matches(sentence, self.entity_set)

    def get_mention_relations(self, sentence: str) -> List[str]:
        """获取问题中提到的关系"""
        return self._find_matches(sentence, self.relation_set)

    def get_mention_attributes(self, sentence: str) -> List[str]:
        """获取问题中提到的属性"""
        return self._find_matches(sentence, self.attribute_set)

    def get_mention_labels(self, sentence: str) -> List[str]:
        """获取问题中提到的标签"""
        return self._find_matches(sentence, self.label_set)

    def _find_matches(self, text: str, pattern_set: Set[str]) -> List[str]:
        """在文本中查找匹配的模式"""
        if not pattern_set:
            return []

        # 按长度降序排序，优先匹配长的模式
        sorted_patterns = sorted(pattern_set, key=len, reverse=True)
        matches = []

        for pattern in sorted_patterns:
            if pattern in text:
                matches.append(pattern)
                # 移除已匹配的部分以避免重复匹配
                text = text.replace(pattern, "", 1)

        return matches

    def parse_sentence(self, sentence: str) -> Dict[str, List[str]]:
        """解析句子，提取实体、关系、标签、属性信息"""
        return {
            "%ENT%": self.get_mention_entities(sentence),
            "%REL%": self.get_mention_relations(sentence),
            "%LAB%": self.get_mention_labels(sentence),
            "%ATT%": self.get_mention_attributes(sentence)
        }

    def decode_value_combination(self, value_combination: Tuple, cypher_check: Dict[str, int]) -> Dict[str, str]:
        """解码值组合"""
        result = {}
        for index, (key, required_count) in enumerate(cypher_check.items()):
            if required_count == 1:
                result[key] = value_combination[index][0]
            else:
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    result[key_num] = value_combination[index][i]
        return result

    def get_combinations(self, cypher_check: Dict[str, int], info: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """获取值组合，限制组合数量避免组合爆炸"""
        slot_values = []

        for key, required_count in cypher_check.items():
            values = info.get(key, [])
            if len(values) < required_count:
                return []

            # 使用组合而不是排列，减少数量
            combinations = list(itertools.combinations(values, required_count))
            if len(combinations) > self.config["max_combinations"]:
                logger.warning(f"组合数量过多({len(combinations)}), 进行截断")
                combinations = combinations[:self.config["max_combinations"]]

            slot_values.append(combinations)

        combinations = []
        for value_combination in itertools.product(*slot_values):
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
            if len(combinations) >= self.config["max_combinations"]:
                break

        return combinations

    def replace_token_in_string(self, string: str, combination: Dict[str, str]) -> str:
        """替换字符串中的令牌"""
        for key, value in combination.items():
            string = string.replace(key, value)
        return string

    def expand_template(self, template: str, cypher: str, cypher_check: Dict[str, int],
                        info: Dict[str, List[str]], answer: str) -> List[Tuple[str, str, str]]:
        """扩展模板"""
        combinations = self.get_combinations(cypher_check, info)
        template_cypher_pairs = []

        for combination in combinations:
            replaced_template = self.replace_token_in_string(template, combination)
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            replaced_answer = self.replace_token_in_string(answer, combination)
            template_cypher_pairs.append((replaced_template, replaced_cypher, replaced_answer))

        return template_cypher_pairs

    def check_cypher_info_valid(self, info: Dict[str, List[str]], cypher_check: Dict[str, int]) -> bool:
        """检查信息是否足够填充模板"""
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    def expand_question_and_cypher(self, info: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
        """扩展问题和Cypher查询"""
        template_cypher_pairs = []

        for template, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                pairs = self.expand_template(template, cypher, cypher_check, info, answer)
                template_cypher_pairs.extend(pairs)

        return template_cypher_pairs

    def sentence_similarity(self, string1: str, string2: str) -> float:
        """计算句子相似度（使用改进的相似度算法）"""
        # 使用编辑距离和Jaccard相似度的组合
        if not string1 or not string2:
            return 0.0

        # 归一化编辑距离
        max_len = max(len(string1), len(string2))
        edit_similarity = 1 - (Levenshtein.distance(string1, string2) / max_len) if max_len > 0 else 0

        # Jaccard相似度
        set1, set2 = set(string1), set(string2)
        jaccard = len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0

        # 组合相似度
        return 0.6 * edit_similarity + 0.4 * jaccard

    def cypher_match(self, sentence: str, info: Dict[str, List[str]]) -> List[Tuple[str, str, float, str]]:
        """匹配Cypher查询"""
        template_cypher_pairs = self.expand_question_and_cypher(info)
        results = []

        for template, cypher, answer in template_cypher_pairs:
            score = self.sentence_similarity(sentence, template)
            if score >= self.config["similarity_threshold"]:
                results.append((template, cypher, score, answer))

        return sorted(results, key=lambda x: x[2], reverse=True)

    def parse_result(self, graph_search_result: List[Dict], answer_template: str, info: Dict[str, List[str]]) -> str:
        """解析查询结果"""
        if not graph_search_result:
            return "抱歉，没有找到相关信息。"

        result = graph_search_result[0]

        # 处理关系类型
        if "REL" in result and hasattr(result["REL"], 'types'):
            result["REL"] = list(result["REL"].types())[0]

        # 替换答案模板中的令牌
        try:
            answer = self.replace_token_in_string(answer_template, result)
            return answer
        except Exception as e:
            logger.error(f"解析结果失败: {e}")
            return "抱歉，处理结果时出现错误。"

    def execute_cypher_safely(self, cypher: str) -> List[Dict]:
        """安全执行Cypher查询"""
        try:
            return self.graph.run(cypher).data()
        except Exception as e:
            logger.error(f"执行Cypher查询失败: {cypher}, 错误: {e}")
            return []

    def query(self, sentence: str) -> str:
        """对外提供问答接口"""
        logger.info(f"处理查询: {sentence}")

        try:
            info = self.parse_sentence(sentence)
            logger.debug(f"解析信息: {info}")

            template_cypher_scores = self.cypher_match(sentence, info)

            for template, cypher, score, answer in template_cypher_scores:
                logger.debug(f"尝试模板: {template} (相似度: {score:.3f})")

                graph_search_result = self.execute_cypher_safely(cypher)

                if graph_search_result:
                    final_answer = self.parse_result(graph_search_result, answer, info)
                    logger.info(f"找到答案: {final_answer}")
                    return final_answer

            return "抱歉，没有找到相关信息。"

        except Exception as e:
            logger.error(f"处理查询时发生错误: {e}")
            return "抱歉，处理问题时出现错误。"


if __name__ == "__main__":
    # 示例用法
    qa_system = GraphQA()

    test_queries = [
        "邓紫棋的歌曲泡沫发行时间",
        "邓紫棋的身高是多少",
        "周杰伦的妻子是谁",
        "李白和杜甫是什么关系"
    ]

    for query in test_queries:
        result = qa_system.query(query)
        print(f"问题: {query}")
        print(f"回答: {result}")
        print("-" * 50)