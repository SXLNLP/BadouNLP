import re
import json
from py2neo import Graph, Node, Relationship
from collections import defaultdict
import logging
from typing import Dict, Set, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化知识图谱构建器

        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        try:
            self.graph = Graph(uri, user=user, password=password)
            self.graph.run("MATCH (n) RETURN n LIMIT 1")  # 测试连接
            logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {e}")
            raise

        self.attribute_data = defaultdict(dict)
        self.relation_data = defaultdict(dict)
        self.label_data = {}
        self.processed_entities = set()
        self.schema_data = defaultdict(set)

    def clean_and_extract_label(self, entity: str) -> str:
        """
        清理实体名称并提取标签

        Args:
            entity: 原始实体名称

        Returns:
            清理后的实体名称
        """
        # 使用更精确的正则表达式匹配中文括号
        pattern = r"（([^（）]+)）|\(([^()]+)\)"
        match = re.search(pattern, entity)

        if match:
            # 提取括号内的内容（支持中文和英文括号）
            label_content = match.group(1) or match.group(2) or ""

            # 检查是否包含预定义的标签
            for label in ["歌曲", "专辑", "歌手"]:
                if label in label_content:
                    self.label_data[entity] = label
                    break

            # 移除括号及其内容
            entity = re.sub(pattern, "", entity)

        return entity.strip()

    def load_triplets(self, relation_file: str, attribute_file: str):
        """
        加载关系三元组和属性三元组

        Args:
            relation_file: 关系三元组文件路径
            attribute_file: 属性三元组文件路径
        """
        # 加载关系三元组
        logger.info("开始加载关系三元组...")
        with open(relation_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        logger.warning(f"第{line_num}行格式错误: {line.strip()}")
                        continue

                    head, relation, tail = parts
                    head = self.clean_and_extract_label(head)
                    tail = self.clean_and_extract_label(tail)

                    self.relation_data[head][relation] = tail
                    self.schema_data["entities"].add(head)
                    self.schema_data["entities"].add(tail)
                    self.schema_data["relations"].add(relation)

                except Exception as e:
                    logger.error(f"处理关系文件第{line_num}行时出错: {e}")

        # 加载属性三元组
        logger.info("开始加载属性三元组...")
        with open(attribute_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split("\t")
                    if len(parts) != 3:
                        logger.warning(f"第{line_num}行格式错误: {line.strip()}")
                        continue

                    entity, attribute, value = parts
                    entity = self.clean_and_extract_label(entity)

                    # 对属性值进行基本清理
                    value = value.strip().replace("'", "\\'").replace('"', '\\"')
                    self.attribute_data[entity][attribute] = value
                    self.schema_data["entities"].add(entity)
                    self.schema_data["attributes"].add(attribute)

                except Exception as e:
                    logger.error(f"处理属性文件第{line_num}行时出错: {e}")

    def build_graph(self, batch_size: int = 1000):
        """
        构建知识图谱

        Args:
            batch_size: 批量处理的大小
        """
        logger.info("开始构建知识图谱...")

        # 使用事务批量处理
        tx = self.graph.begin()
        processed_count = 0

        # 处理所有实体（包括有属性和只有关系的）
        all_entities = set(self.attribute_data.keys()) | set(self.relation_data.keys())
        all_entities.update([tail for relations in self.relation_data.values()
                             for tail in relations.values()])

        # 创建节点
        nodes_dict = {}
        for entity in all_entities:
            if entity not in self.processed_entities:
                # 准备属性
                properties = {"NAME": entity}
                if entity in self.attribute_data:
                    properties.update(self.attribute_data[entity])

                # 确定标签
                labels = ["Entity"]  # 默认标签
                if entity in self.label_data:
                    labels.append(self.label_data[entity])

                # 创建节点
                node = Node(*labels, **properties)
                nodes_dict[entity] = node
                tx.create(node)
                self.processed_entities.add(entity)

                processed_count += 1
                if processed_count % batch_size == 0:
                    tx.commit()
                    logger.info(f"已处理 {processed_count} 个节点")
                    tx = self.graph.begin()

        # 创建关系
        relation_count = 0
        for head, relations in self.relation_data.items():
            for relation_type, tail in relations.items():
                if head in nodes_dict and tail in nodes_dict:
                    relationship = Relationship(nodes_dict[head], relation_type, nodes_dict[tail])
                    tx.create(relationship)
                    relation_count += 1

                    if relation_count % batch_size == 0:
                        tx.commit()
                        logger.info(f"已处理 {relation_count} 个关系")
                        tx = self.graph.begin()

        # 提交剩余的事务
        tx.commit()
        logger.info(f"知识图谱构建完成！共创建 {len(nodes_dict)} 个节点和 {relation_count} 个关系")

    def save_schema(self, output_file: str):
        """
        保存知识图谱模式信息

        Args:
            output_file: 输出文件路径
        """
        # 添加标签信息
        for label in set(self.label_data.values()):
            self.schema_data["labels"].add(label)

        # 转换为列表格式
        schema_dict = {key: list(value) for key, value in self.schema_data.items()}

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(schema_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"模式信息已保存到 {output_file}")
        except Exception as e:
            logger.error(f"保存模式信息失败: {e}")

    def clear_existing_data(self):
        """清空现有图数据"""
        logger.warning("正在清空现有图数据...")
        self.graph.run("MATCH (n) DETACH DELETE n")
        logger.info("图数据已清空")


def main():
    # 配置参数
    NEO4J_URI = "bolt://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "yzyzzz999"
    RELATION_FILE = "triplets_head_rel_tail.txt"
    ATTRIBUTE_FILE = "triplets_enti_attr_value.txt"
    SCHEMA_FILE = "kg_schema.json"

    try:
        # 初始化构建器
        kg_builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # 清空现有数据（可选，根据需求决定是否启用）
        # kg_builder.clear_existing_data()

        # 加载数据
        kg_builder.load_triplets(RELATION_FILE, ATTRIBUTE_FILE)

        # 构建图谱
        kg_builder.build_graph(batch_size=500)

        # 保存模式信息
        kg_builder.save_schema(SCHEMA_FILE)

        logger.info("知识图谱构建流程完成！")

    except Exception as e:
        logger.error(f"知识图谱构建失败: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())