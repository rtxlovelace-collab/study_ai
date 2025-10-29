'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 8_cohere_rerank.py
* @Time: 2025/9/10
* @All Rights Reserve By Brtc
'''
import os

import dotenv
import weaviate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
rerank = CohereRerank(model="rerank-multilingual-v3.0")
client = weaviate.connect_to_weaviate_cloud(
    skip_init_checks=True,
    cluster_url=os.getenv("WAEVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
db = WeaviateVectorStore(client=client,
                         index_name="DataSetTest",
                         text_key="text",
                         embedding=embedding)
# 构建压缩检索器
retriever =  ContextualCompressionRetriever(
    base_retriever=db.as_retriever(),
    base_compressor=rerank,
)

# 执行搜索并排序
search_docs = retriever.invoke("关于LLMOPS的应用配置信息有那些？")
print(search_docs)
print(len(search_docs))
client.close()
