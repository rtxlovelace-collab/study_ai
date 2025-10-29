'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 7_study_faiss_db.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_texts([
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
], embedding, relevance_score_fn = lambda  distance:1.0/(1.0 + distance))

print(db.index.ntotal)
res = db.similarity_search_with_score("我养了一只猫，叫笨笨")
for one in res:
    print(one)
print("=======================================================================")
res = db.similarity_search_with_relevance_scores("我养了一只猫，叫笨笨")
for one in res:
    print(one)
