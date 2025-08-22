import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import  InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
import dashscope
from http import HTTPStatus
import time
class MeiTuanRag:
    def __init__(self):
        self.main()

    def main(self):
        self.compentents_loader()
        self.cutdocs()
        self.embed()
        self.revetrial()
        self.convertion()

    def compentents_loader(self):
        loader=TextLoader("knowledge.txt",encoding="utf-8")
        self.documents=loader.load()
        self.llm=ChatOpenAI(
            model="glm4:9b",
            base_url="http://localhost:11434/v1",  # 智谱API兼容地址
            openai_api_key="1234",
            stream=True
        )
        # self.llm=ChatOpenAI(
        #     api_key=os.environ['zhipuai_api_key'],
        #     base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱API兼容地址
        #     model="glm-4",
        #     stream=True
        # )
        self.embeddings=OpenAIEmbeddings(
            api_key=os.environ['zhipuai_api_key'],
            base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱API兼容地址
            model="embedding-2"
        )
        self.prompt=ChatPromptTemplate.from_messages([
            ("system","你是一名美团外卖app智能客服,需要结合历史对话和检索到的知识库回答,\n知识：{context}\n历史:{history},请你直接返回和我的问题对应的知识库中的整条答案,不要删减或自己加东西,如果有不太清楚但是知识库比较相关的内容，直接用知识库内容就好,如果有一个问题对应多种情况的答案,没有特殊说明的话输出多种情况,如果只问到某一种或者某几种答案的话,输出对应几种情况的答案,如果完全不相关，输出我不知道"),
            ("user","{input}")
        ])
        self.history=InMemoryChatMessageHistory()

    def cutdocs(self):
        flag=0
        if flag==0:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,  # 适当增大片段长度（根据知识库内容调整）
                chunk_overlap=0,  # 保留部分重叠，避免切割语义
                # separators=["\n\n", "\n", "。", "，", " "]  # 按优先级分割，更符合中文习惯
            )
            self.segements = text_splitter.split_documents(self.documents)
        #文本分割
        else:
            text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=200,chunk_overlap=0)
            self.segements=text_splitter.split_documents(self.documents)
        texts=[chunk.page_content for chunk in self.segements]
        # print(f"分割后的片段长度为{len(texts)}")
        # for document in texts:
        #     print(document)
        #     print("--------------------------------")

    def embed(self):
        # 正确步骤：先删除旧集合，再创建新集合，最后查看内容
        from chromadb import PersistentClient
        # 1. 连接到持久化目录并删除旧集合（关键：在创建新集合前执行）
        client = PersistentClient(path="../chroma.db")
        if "langchain" in [col.name for col in client.list_collections()]:
            client.delete_collection(name="langchain")
            # print("已删除旧集合")

        # 2. 重新创建向量库（此时集合为空，只会添加当前的segements）
        self.vector = Chroma.from_documents(
            self.segements,
            self.embeddings,
            persist_directory="../chroma.db"
        )
        # print(f"已添加新片段，数量：{len(self.segements)}")

        # 3. 查看向量库内容
        client = PersistentClient(path="../chroma.db")
        collection = client.get_collection(name="langchain")  # 获取刚创建的集合
        all_docs = collection.get(limit=100)
        # print(f"向量库中实际存储的片段数量: {len(all_docs['ids'])}")
        # for i in range(len(all_docs['ids'])):
        #     print(f"\n片段ID: {all_docs['ids'][i]}")
        #     print(f"内容: {all_docs['documents'][i]}")
        #     print(f"元数据: {all_docs['metadatas'][i]}")
        #     print("-" * 50)

    def revetrial(self):
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.retrieval = self.vector.as_retriever(
            search_type="mmr",  # 最大化边际相关性
            search_kwargs={
                "k": 15,
                "lambda_mult": 0.1  # 平衡相关性和多样性
            }
        )

    def text_rerank(self,query, candidate_docs):
        # 提取候选文档内容
        docs = [d.page_content for d in candidate_docs]
        # 调用重排序API
        resp = dashscope.TextReRank.call(
            model="gte-rerank-v2",
            query=query,
            documents=docs,
            top_n=6,
            api_key=os.environ.get("aliyunbailian_api_key")
        )

        # 强制用候选文档内容兜底（忽略API返回的null）
        if resp.status_code == HTTPStatus.OK and "results" in resp.output:
            reranked = []
            for item in resp.output["results"]:
                # 直接使用候选文档的原始内容（因为API返回null）
                original_doc = candidate_docs[item["index"]]
                reranked.append(Document(
                    page_content=original_doc.page_content,  # 用候选文档内容
                    metadata=original_doc.metadata
                ))
            return reranked[:6]
        else:
            # API调用失败时，直接返回前3个候选文档
            return [Document(page_content=d.page_content, metadata=d.metadata) for d in candidate_docs[:3]]

    def safe_stream_generate(self, input_data, max_retries=3):
        retry_count = 0
        chunk_index = 0  # 序号标记，保证顺序
        first_try=True
        while retry_count < max_retries:
            try:
                stream = self.document_chain.stream(input_data)
                for chunk in stream:
                    if first_try and chunk_index == 3:  # 生成到第2个chunk时故意报错
                        first_try = False
                        raise Exception("模拟网络中断")
                    yield (chunk_index, chunk)  # 返回(序号, 内容)
                    chunk_index += 1
                break
            except Exception as e:
                retry_count += 1
                print(f"重试({retry_count}/{max_retries})：{e}")
                if retry_count >= max_retries:
                    yield (-1, "[内容传输中断，请重试]")  # 失败标记
                time.sleep(1)

    def convertion(self):
        while True:
            user_input = input("请提出你的问题:")
            print("思考中-------")
            if not user_input:
                print("输入不能为空，请重新输入。")
                continue  # 空输入时跳过后续处理，直接进入下一轮

            #粗排
            start_time=time.time()
            candidate_docs=self.retrieval.invoke(user_input)
            middle_time=time.time()
            flag=1
            if flag==0:
                for index,candidate_doc in enumerate(candidate_docs):
                    print(f"粗排第{index+1}条")
                    print(candidate_doc.page_content)
                    print("___________________")
            # print(f"粗排耗时{round((middle_time-start_time)/60,2)}")

            #精排
            middle_time=time.time()
            reranked_docs=self.text_rerank(user_input,candidate_docs)
            end_time=time.time()
            if flag==0:
                for index,text in enumerate([doc.page_content for doc in reranked_docs]):
                    print(f"重排序后的第{index+1}条:")
                    print(text)
                    print("______")
            # print(f"精排耗时{round((end_time-middle_time)/60,2)}")

            #执行链条，流式输出
            input_data={"input":user_input,
                        "history":self.history,
                        "context":reranked_docs
                        }
            # response = self.document_chain.stream(input_data)
            response_full = ""
            print("回答：", end="", flush=True)
            for chunk_index, chunk in self.safe_stream_generate(input_data):
                if chunk_index == -1:
                    print(chunk, end="", flush=True)  # 输出错误提示
                else:
                    print(chunk, end="", flush=True)  # 正常输出内容
                response_full += chunk
            print("")
            #结束问答，保存该轮对话内容
            # print("回答完毕")
            self.history.add_message(HumanMessage(content=user_input))
            self.history.add_message(AIMessage(content=response_full))

if __name__ == "__main__":
    rag=MeiTuanRag()
    # print(len([chunk.page_content for chunk in self.segements]))
