import dotenv
import os

import dotenv
from langchain.tools import retriever

import weaviate
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey
dotenv.load_dotenv()

#构建向量数据库
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
db = WeaviateVectorStore(
    client=client,
    embedding=embedding,
    index_name = "DataSetTest",
    text_key = "text",)

#执行相似性检索，并返回k条数据
# retriever = db.as_retriever(
#     search_type = "similarity_score_threshold",
#     search_kwargs = {"k": 10, "similarity_threshold": 0.5}
# )
resutls = retriever.invoke("关于应用配置的接口有哪些？")
for one in resutls:
    print("="*40)
    print(one.page_content[:30])
client.close()
