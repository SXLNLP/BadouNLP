
import json
import os
from datetime import datetime, timedelta
import time
import jieba
import numpy as np
# from zai import ZhipuAiClient
from zhipuai import ZhipuAI
from bm25 import BM25
from job_assistant.config import StorageConfig
from job_assistant.storage.pojo import DataPojo

'''
基于RAG来介绍工作职位信息。职位信息是来源自招聘网站的抓取的数据
用bm25做召回
同样以来智谱的api作为我们的大模型
https://docs.bigmodel.cn/cn/guide/start/model-overview
'''

#智谱的api作为我们的大模型
def call_large_model(prompt):
    # client = ZhipuAiClient(api_key=os.environ.get("zhipuApiKey")) # 填写您自己的APIKey
    client = ZhipuAI(api_key=os.environ.get("zhipuApiKey"))  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4.5",  # 填写需要调用的模型名称
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    response_text = response.choices[0].message.content
    return response_text

class SimpleRAG:
    def __init__(self, folder_path="../data"):
        # self.load_hero_data_from_localtxt(folder_path)
        self.load_data_from_db()

    #从本地txt文件加载知识库数据
    def load_hero_data_from_localtxt(self, folder_path):
        self.hero_data = {}
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as file:
                    intro = file.read()
                    hero = file_name.split(".")[0]
                    self.hero_data[hero] = intro
        corpus = {}
        self.index_to_name = {}
        index = 0
        for hero, intro in self.hero_data.items():
            corpus[hero] = jieba.lcut(intro)
            self.index_to_name[index] = hero
            index += 1
        self.bm25_model = BM25(corpus)
        return

    #从数据库加载知识库数据
    def load_data_from_db(self):
        self.hero_data = {}

        data_pojo = DataPojo(StorageConfig)

        #今天跟明天的时间
        today = datetime.now()
        tommorow = today + timedelta(days=1)
        # 将日期转成yyyy-mm-dd日期字符串
        date_today_str = today.strftime("%Y-%m-%d")
        date_tommorow_str = tommorow.strftime("%Y-%m-%d")
        # 将yyyy-mm-dd日期字符串转成datetime对象时间抽
        date_today_object = datetime.strptime(date_today_str, "%Y-%m-%d")
        date_tommorow_object = datetime.strptime(date_tommorow_str, "%Y-%m-%d")
        # 将datetime对象转成时间戳
        date_today_timestamp = int(time.mktime(date_today_object.timetuple()))
        date_tommorow_timestamp = int(time.mktime(date_tommorow_object.timetuple()))

        jobs = data_pojo.select_joblist_by_date(date_today_timestamp, date_tommorow_timestamp)
        for job in jobs:
            self.hero_data[job.job_id] = job.job_description

        corpus = {}
        self.index_to_name = {}
        index = 0
        for id, text in self.hero_data.items():
            corpus[id] = jieba.lcut(text)
            self.index_to_name[index] = id
            index += 1
        self.bm25_model = BM25(corpus)
        return

    def retrive(self, user_query):
        scores = self.bm25_model.get_scores(jieba.lcut(user_query))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        hero = sorted_scores[0][0]
        text = self.hero_data[hero]
        return text

    def query(self, user_query):    
        print("user_query:", user_query)
        print("=======================")
        retrive_text = self.retrive(user_query)
        print("retrive_text:", retrive_text)
        print("=======================")
        prompt = f"请根据以下从数据库中获得的职位详情，回答用户问题：\n\n职位详情介绍：\n{retrive_text}\n\n用户问题：{user_query}"
        response_text = call_large_model(prompt)
        print("模型回答：", response_text)
        print("=======================")

if __name__ == "__main__":
    rag = SimpleRAG()
    user_query = "我擅长大模型微调，工作中日常会有哪些工作，以及要掌握哪些技能？"
    rag.query(user_query)

    print("----------------")
    print("No RAG (直接请求大模型回答)：")
    print(call_large_model(user_query))
