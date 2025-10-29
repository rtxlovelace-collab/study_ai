'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 1_study_mult_query_search.py
* @Time: 2025/9/9
* @All Rights Reserve By Brtc
'''
import os

import dotenv
import weaviate
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()
# 构建向量数据库
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WAEVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = WeaviateVectorStore(client=client,
                         index_name="DataSetTest",
                         text_key="text",
                         embedding=embedding)

retriever = db.as_retriever(search_type = "mmr")
#创建多重查询检索器
mult_query = MultiQueryRetriever.from_llm(
    retriever = retriever,
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0),
    prompt=ChatPromptTemplate.from_template(
        "你是一个AI语言模型助手。你的任务是生成给定用户问题的3个不同版本，以从向量数据库中检索相关文档。"
        "通过提供用户问题的多个视角，你的目标是帮助用户克服基于距离的相似性搜索的一些限制。"
        "请用换行符分隔这些替代问题。"
        "原始问题：{question}"
    )
)
#执行检索
docs = mult_query.invoke("关于LLMops的应用配置文件有哪些？")
for rank, doc in enumerate(docs):
    print("==========================================================")
    print(doc.page_content[:30])
client.close()
