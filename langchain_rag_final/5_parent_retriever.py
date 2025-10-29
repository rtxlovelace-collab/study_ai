'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_parent_retriever.py
* @Time: 2025/9/10
* @All Rights Reserve By Brtc
'''
import os

import dotenv
import weaviate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()

# 1、创建加载器和文档列表， 并加载 文档
loaders = [
    UnstructuredFileLoader("./eshop_goods.txt"),
    UnstructuredFileLoader("./01.项目API文档.md")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
#2、创建文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
#3、创建向量数据库与文档数据库
#3、构建向量数据库与检索器
client = weaviate.connect_to_weaviate_cloud(
    skip_init_checks=True,
    cluster_url=os.getenv("WAEVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = WeaviateVectorStore(client=client,
                         index_name="ParentTest",
                         text_key="text",
                         embedding=embedding)
store = LocalFileStore("./parent-documents")

# 4、创建父文档检索器
retriever = ParentDocumentRetriever(
    vectorstore=db,
    byte_store=store,
    child_splitter=text_splitter
)

#5、添加为文档
#retriever.add_documents(docs, ids=None)

#6、检索名返回内容
search_docs = retriever.invoke("分享一些关于LLMops的应用配置")
for one in search_docs:
    print("=============================================")
    print(one)
client.close()
