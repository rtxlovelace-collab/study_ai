import os

import dotenv
import weaviate
from langchain import text_splitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from openai import vector_stores
from weaviate.auth import AuthApiKey

from 作业.尝试 import client

dotenv.load_dotenv()

#创建加载器与文档列表，并加载文档
loaders = [
    UnstructuredFileLoader("./数据文档/eshop_goods.txt"),
    UnstructuredFileLoader("./数据文档/01.项目API文档.md")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# 2.创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,#指定每个文本片段的最大长度为500个字符
        chunk_overlap=50,#指定文本片段的重叠长度为50个字符
    )


#创建向量数据库与文档数据库
vector_store = WeaviateVectorStore(
    client=weaviate.connect_to_weaviate_cloud(#建立与云服务的连接
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
    ),
    index_name="study",#指定在 Weaviate 中操作的数据集合（类）名称为 "study"
    text_key="metadata",#指定存储原始文本数据的字段名为 "metadata"
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)
print(text_splitter)
#将文档数据库
store = LocalFileStore("./parent-document")

#创建父文档检索器
retriver = ParentDocumentRetriever(
    vectorstore=vector_store,#指定存储子文档向量的向量数据库，
    byte_store=store,#指定存储原始父文档数据的字节存储
    child_splitter=text_splitter,#指定子文档的文本分割器

)
client.close()
#添加文档
retriver.add_documents(docs,ids=None)

#检索并返回内容
search_docs = retriver.invoke("分享关于LLMOps的一些相关应用配置")
print(search_docs)
print(len(search_docs))

