'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 11_study_pincone_filter.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

dotenv.load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = PineconeVectorStore(index_name="for-study", embedding=embedding, namespace="DamoTest")
# 单条件检索
#search = db.similarity_search_with_score("我养了一只猫，叫笨笨", namespace="DamoTest", filter={"page":{"$lte":5}})

# 多条件
search = db.similarity_search_with_score("我养了一只猫，叫笨笨",
                                         namespace="DamoTest",
                                         filter={"$and":[{"page":5}, {"account_id":9527}]})
