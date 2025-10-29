#FAISS 是一个开源的高效的向量搜索库，可以用于大规模向量搜索。

import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


dotenv.load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#FAISS 是一个开源的高效的向量搜索库，可以用于大规模向量搜索。
#这里我们使用LangChain 封装的 OpenAI 文本嵌入模型来生成向量。
#开始存入到faiss中
db=FAISS.from_texts([ "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",], embeddings,
                    relevance_score_fn=lambda distance: 1.0/(1.0+distance))
#打印一共存了多少条数据
print(db.index.ntotal)
#欧几里得距离进行相似度计算
res = db.similarity_search_with_score("我养了一只叫笨笨的猫")
for one in res:
    print(one)
print("="*40)
'''
#由于同文本嵌入模型生成向量的范围不一致，LangChain 封装的 Faiss 计算相关性得分的时候，
可能会出现 bug（比如出现负数）,所以需要进行修正 1.0 / (1.0 + distance)
'''
res = db.similarity_search_with_relevance_scores("我养了一只叫笨笨的猫")
for one in res:
    print(one)