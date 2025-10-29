'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 21_study_retriever.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
import os
import dotenv
import weaviate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()
#1、构建文档分割器
loader = UnstructuredMarkdownLoader("./01.项目API文档.md")
#2、构建分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。|！|？", "\.\s|\!\s|\?\s", "；|;\s", "，|,\s", " ", "", ],
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True
)
documents = loader.load()
chunks = text_splitter.split_documents(documents)
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

# 检索
#results =db.max_marginal_relevance_search("关于应用配置的接口有那些？")
results = db.similarity_search("关于应用配置的接口有那些？")

for one in results:
    print("===========================================")
    print(one.page_content[:30])
client.close()


