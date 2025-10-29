'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 12_study_pinecone_crud.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

dotenv.load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = PineconeVectorStore(index_name="brtc2online", embedding=embedding, namespace="brtcDataset")

# 删除id
db.delete(ids=["6bb58f6d-930e-4019-a6d7-a266766cbb0a"])

# 更新数据
pinecone_index = db.get_pinecone_index("brtc2online")
pinecone_index.update(id="xxx", values=[], set_metadata=[], namespace="space")

