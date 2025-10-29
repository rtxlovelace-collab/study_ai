'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 9_study_faiss_crud.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
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
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_texts(texts, embedding, metadatas)

#1、数据删除操作
print("数据删除前:",db.index.ntotal)
db.delete([db.index_to_docstore_id[0]])
print("删除后的数据:", db.index.ntotal)

#2、数据存储到本地
db.save_local("./faiss-store/")

#3、从本地加载数据
newdb = FAISS.load_local("./faiss-store/", embeddings=embedding, allow_dangerous_deserialization=True)
docs = newdb.similarity_search_with_score("笨笨喜欢睡觉吗？")
for one in docs:
    print(one)