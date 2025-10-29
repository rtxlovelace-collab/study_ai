'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 23_study_vectore_retriever.py
* @Time: 2025/9/9
* @All Rights Reserve By Brtc
'''
import os

import dotenv
import weaviate
from langchain_core.runnables import ConfigurableField
from langchain_openai import OpenAIEmbeddings
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

# 2、转换器检索
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.5}
).configurable_fields(
    search_type = ConfigurableField(id="db_search_type"),
    search_kwargs = ConfigurableField(id="db_search_kwargs"),
)

#执行相似性检索，并返回K条数据
"""
results = retriever.with_config(configurable={
    "db_search_type":"mmr",
    "db_search_kwargs":{"k":4}
}).invoke("关于应用配置的接口有那些？")
"""
results = retriever.invoke("关于应用配置的接口有那些？")
for one in results:
    print("===========================================")
    print(one.page_content[:30])
client.close()
