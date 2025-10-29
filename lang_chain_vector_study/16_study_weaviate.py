'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 16_study_weaviate.py
* @Time: 2025/9/5
* @All Rights Reserve By Brtc
'''
import os

import dotenv
import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.collections.classes.filters import Filter

texts = [
    "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",
]
metadatas: list = [
    {"page": 1},
    {"page": 2},
    {"page": 3},
    {"page": 4},
    {"page": 5},
    {"page": 6},
    {"page": 7},
    {"page": 8},
    {"page": 9},
    {"page": 10},
]
dotenv.load_dotenv()
'''
client = weaviate.connect_to_cloud(
    cluster_url=os.getenv("WAEVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
'''

client = weaviate.connect_to_local(os.getenv("WEAVIATE_HOST"))
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = WeaviateVectorStore(client=client, index_name="DataSetTest", text_key="text",embedding=embedding)

#ids = db.add_texts(texts = texts, metadatas = metadatas)
#print(ids)

#search_test = db.similarity_search_with_score("笨笨")
#带国过滤性搜索
filter = Filter.by_property("page").greater_than(5)
search_test = db.similarity_search_with_score("没电了", filters=filter)
for one in search_test:
    print(one)
client.close()
