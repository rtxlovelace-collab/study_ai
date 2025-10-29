import os

import dotenv
import weaviate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey
from langchain_cohere import CohereRerank
dotenv.load_dotenv()

# 1.构建向量模型
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

rerank = CohereRerank(model="rerank-multilingual-v3.0")
client = weaviate.connect_to_weaviate_cloud(
    skip_init_checks=True,
    cluster_url=os.getenv("WEAVIATE_HOST"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_PORT"))
)

db = WeaviateVectorStore(
    client=client,
    embedding=embedding,
    index_name="cohere-index",
    text_key="content",

)
# 2.构建压缩检索器
retriever = ContextualCompressionRetriever(
    base_retriever=db.as_retriever(),
    base_compressor=rerank,
)

# 3.执行搜索并排序
search_docs = retriever.invoke("关于LLMOps应用配置的信息有哪些呢？")
print(search_docs)
print(len(search_docs))
retriever = ContextualCompressionRetriever(
    base_retriever=db.as_retriever(),
    base_compressor=rerank,
)

# 3.执行搜索并排序

client.close()