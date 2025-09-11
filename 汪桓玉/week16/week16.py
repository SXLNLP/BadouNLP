#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neo4j知识图谱问答系统
实现基于Neo4j的简单知识图谱问答功能
"""

import json
import re
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional

class Neo4jKnowledgeGraph:
    """Neo4j知识图谱管理类"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        """初始化Neo4j连接"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            
    def clear_database(self):
        """清空数据库"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("数据库已清空")
            
    def create_sample_data(self):
        """创建示例知识图谱数据"""
        with self.driver.session() as session:
            # 创建电影节点
            movies = [
                {"title": "泰坦尼克号", "year": 1997, "genre": "爱情片"},
                {"title": "阿凡达", "year": 2009, "genre": "科幻片"},
                {"title": "盗梦空间", "year": 2010, "genre": "科幻片"},
                {"title": "肖申克的救赎", "year": 1994, "genre": "剧情片"},
                {"title": "教父", "year": 1972, "genre": "犯罪片"}
            ]
            
            for movie in movies:
                session.run(
                    "CREATE (m:Movie {title: $title, year: $year, genre: $genre})",
                    title=movie["title"], year=movie["year"], genre=movie["genre"]
                )
                
            # 创建演员节点
            actors = [
                {"name": "莱昂纳多·迪卡普里奥", "birth_year": 1974, "nationality": "美国"},
                {"name": "凯特·温斯莱特", "birth_year": 1975, "nationality": "英国"},
                {"name": "萨姆·沃辛顿", "birth_year": 1976, "nationality": "澳大利亚"},
                {"name": "佐伊·索尔达娜", "birth_year": 1978, "nationality": "美国"},
                {"name": "蒂姆·罗宾斯", "birth_year": 1958, "nationality": "美国"},
                {"name": "摩根·弗里曼", "birth_year": 1937, "nationality": "美国"},
                {"name": "马龙·白兰度", "birth_year": 1924, "nationality": "美国"},
                {"name": "阿尔·帕西诺", "birth_year": 1940, "nationality": "美国"}
            ]
            
            for actor in actors:
                session.run(
                    "CREATE (a:Actor {name: $name, birth_year: $birth_year, nationality: $nationality})",
                    name=actor["name"], birth_year=actor["birth_year"], nationality=actor["nationality"]
                )
                
            # 创建导演节点
            directors = [
                {"name": "詹姆斯·卡梅隆", "birth_year": 1954, "nationality": "加拿大"},
                {"name": "克里斯托弗·诺兰", "birth_year": 1970, "nationality": "英国"},
                {"name": "弗兰克·德拉邦特", "birth_year": 1959, "nationality": "法国"},
                {"name": "弗朗西斯·福特·科波拉", "birth_year": 1939, "nationality": "美国"}
            ]
            
            for director in directors:
                session.run(
                    "CREATE (d:Director {name: $name, birth_year: $birth_year, nationality: $nationality})",
                    name=director["name"], birth_year=director["birth_year"], nationality=director["nationality"]
                )
                
            # 创建关系
            relationships = [
                # 演员-电影关系
                ("莱昂纳多·迪卡普里奥", "ACTED_IN", "泰坦尼克号"),
                ("莱昂纳多·迪卡普里奥", "ACTED_IN", "盗梦空间"),
                ("凯特·温斯莱特", "ACTED_IN", "泰坦尼克号"),
                ("萨姆·沃辛顿", "ACTED_IN", "阿凡达"),
                ("佐伊·索尔达娜", "ACTED_IN", "阿凡达"),
                ("蒂姆·罗宾斯", "ACTED_IN", "肖申克的救赎"),
                ("摩根·弗里曼", "ACTED_IN", "肖申克的救赎"),
                ("马龙·白兰度", "ACTED_IN", "教父"),
                ("阿尔·帕西诺", "ACTED_IN", "教父"),
                
                # 导演-电影关系
                ("詹姆斯·卡梅隆", "DIRECTED", "泰坦尼克号"),
                ("詹姆斯·卡梅隆", "DIRECTED", "阿凡达"),
                ("克里斯托弗·诺兰", "DIRECTED", "盗梦空间"),
                ("弗兰克·德拉邦特", "DIRECTED", "肖申克的救赎"),
                ("弗朗西斯·福特·科波拉", "DIRECTED", "教父")
            ]
            
            for person, relation, movie in relationships:
                if relation == "ACTED_IN":
                    session.run(
                        "MATCH (a:Actor {name: $person}), (m:Movie {title: $movie}) "
                        "CREATE (a)-[:ACTED_IN]->(m)",
                        person=person, movie=movie
                    )
                elif relation == "DIRECTED":
                    session.run(
                        "MATCH (d:Director {name: $person}), (m:Movie {title: $movie}) "
                        "CREATE (d)-[:DIRECTED]->(m)",
                        person=person, movie=movie
                    )
                    
            print("示例数据创建完成")
            
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """执行Cypher查询"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

class QuestionAnswering:
    """问答系统类"""
    
    def __init__(self, kg: Neo4jKnowledgeGraph):
        self.kg = kg
        self.question_patterns = {
            "actor_movies": {
                "pattern": r"(.+?)演了什么电影|(.+?)出演了什么电影|(.+?)的电影有哪些",
                "query": "MATCH (a:Actor {name: $name})-[:ACTED_IN]->(m:Movie) RETURN m.title as movie"
            },
            "movie_actors": {
                "pattern": r"(.+?)的演员有谁|谁演了(.+?)|(.+?)的主演是谁",
                "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie {title: $title}) RETURN a.name as actor"
            },
            "director_movies": {
                "pattern": r"(.+?)导演了什么电影|(.+?)的作品有哪些",
                "query": "MATCH (d:Director {name: $name})-[:DIRECTED]->(m:Movie) RETURN m.title as movie"
            },
            "movie_director": {
                "pattern": r"(.+?)的导演是谁|谁导演了(.+?)",
                "query": "MATCH (d:Director)-[:DIRECTED]->(m:Movie {title: $title}) RETURN d.name as director"
            },
            "movie_info": {
                "pattern": r"(.+?)是什么类型的电影|(.+?)的信息",
                "query": "MATCH (m:Movie {title: $title}) RETURN m.title as title, m.year as year, m.genre as genre"
            },
            "actor_info": {
                "pattern": r"(.+?)的个人信息|(.+?)是哪国人",
                "query": "MATCH (a:Actor {name: $name}) RETURN a.name as name, a.birth_year as birth_year, a.nationality as nationality"
            }
        }
        
    def parse_question(self, question: str) -> Optional[Dict]:
        """解析问题并返回查询信息"""
        for question_type, config in self.question_patterns.items():
            match = re.search(config["pattern"], question)
            if match:
                # 提取匹配的实体名称
                entity = None
                for group in match.groups():
                    if group and group.strip():
                        entity = group.strip()
                        break
                        
                if entity:
                    return {
                        "type": question_type,
                        "entity": entity,
                        "query": config["query"]
                    }
        return None
        
    def answer_question(self, question: str) -> str:
        """回答问题"""
        parsed = self.parse_question(question)
        if not parsed:
            return "抱歉，我无法理解您的问题。请尝试问一些关于电影、演员或导演的问题。"
            
        entity = parsed["entity"]
        query = parsed["query"]
        question_type = parsed["type"]
        
        try:
            # 根据问题类型设置参数
            if "actor" in question_type or "director" in question_type:
                if "movie" in question_type:
                    # 查询某人的电影
                    results = self.kg.execute_query(query, {"name": entity})
                else:
                    # 查询某人的信息
                    results = self.kg.execute_query(query, {"name": entity})
            else:
                # 查询电影相关信息
                results = self.kg.execute_query(query, {"title": entity})
                
            if not results:
                return f"抱歉，我没有找到关于'{entity}'的相关信息。"
                
            # 格式化答案
            return self.format_answer(question_type, entity, results)
            
        except Exception as e:
            return f"查询时出现错误：{str(e)}"
            
    def format_answer(self, question_type: str, entity: str, results: List[Dict]) -> str:
        """格式化答案"""
        if question_type == "actor_movies":
            movies = [result["movie"] for result in results]
            return f"{entity}出演的电影有：{', '.join(movies)}"
            
        elif question_type == "movie_actors":
            actors = [result["actor"] for result in results]
            return f"《{entity}》的演员有：{', '.join(actors)}"
            
        elif question_type == "director_movies":
            movies = [result["movie"] for result in results]
            return f"{entity}导演的电影有：{', '.join(movies)}"
            
        elif question_type == "movie_director":
            directors = [result["director"] for result in results]
            return f"《{entity}》的导演是：{', '.join(directors)}"
            
        elif question_type == "movie_info":
            info = results[0]
            return f"《{info['title']}》是{info['year']}年的{info['genre']}"
            
        elif question_type == "actor_info":
            info = results[0]
            return f"{info['name']}出生于{info['birth_year']}年，国籍是{info['nationality']}"
            
        return "无法格式化答案"

def main():
    """主函数"""
    print("=== Neo4j知识图谱问答系统 ===")
    print("正在连接Neo4j数据库...")
    
    # 初始化知识图谱
    try:
        kg = Neo4jKnowledgeGraph()
        print("数据库连接成功！")
        
        # 清空并创建示例数据
        print("正在初始化数据...")
        kg.clear_database()
        kg.create_sample_data()
        
        # 初始化问答系统
        qa = QuestionAnswering(kg)
        
        print("\n系统初始化完成！您可以开始提问了。")
        print("示例问题：")
        print("- 莱昂纳多·迪卡普里奥演了什么电影？")
        print("- 泰坦尼克号的演员有谁？")
        print("- 詹姆斯·卡梅隆导演了什么电影？")
        print("- 盗梦空间的导演是谁？")
        print("- 阿凡达是什么类型的电影？")
        print("- 输入'退出'结束程序\n")
        
        # 交互式问答
        while True:
            question = input("请输入您的问题：").strip()
            
            if question in ['退出', 'quit', 'exit', 'q']:
                break
                
            if not question:
                continue
                
            answer = qa.answer_question(question)
            print(f"答案：{answer}\n")
            
    except Exception as e:
        print(f"系统初始化失败：{str(e)}")
        print("请确保Neo4j数据库已启动，并检查连接配置。")
        
    finally:
        try:
            kg.close()
            print("数据库连接已关闭。")
        except:
            pass

if __name__ == "__main__":
    main()

