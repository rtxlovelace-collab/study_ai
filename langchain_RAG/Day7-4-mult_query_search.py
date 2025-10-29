import os

import dotenv
import weaviate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey
from langchain.retrievers import MultiQueryRetriever

'''以 weaviate 向量数据库作为检索器，使用 多查询策略 优化普通的 RAG检索，提升检索效率'''
dotenv.load_dotenv()
#构建向量库
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)

embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
db = WeaviateVectorStore(client= client,
                         index_name="DataSetTest",
                         text_key="text",
                         embedding=embedding)

retriever = db.as_retriever(search_type="mmr")
#创建多重查询检索器
mult_query =MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
)
'''MultiQueryRetriever（多查询检索器）是大语言模型（LLM）领域中用于提升信息检索准确性的核心工具，
本质是通过 “多轮衍生查询 + 融合结果” 的逻辑，
解决传统单查询检索中 “查询表述不精准导致漏检 / 误检” 的问题，常见于 RAG（检索增强生成）等需要结合外部知识库的场景。'''

#执行检索
docs= mult_query.invoke("关于LLMOps应用配置的文档有哪些")
print(docs)
print(len(docs))
client.close()
