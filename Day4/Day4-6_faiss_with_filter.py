import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


dotenv.load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#FAISS 是一个开源的高效的向量搜索库，可以用于大规模向量搜索。
#这里我们使用FAISS来存储和检索文本向量。
#开始存入到faiss中
texts=([ "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",]
                    )

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
db=FAISS.from_texts(texts,embeddings,metadatas)
print(db.index_to_docstore_id)
res = db.similarity_search_with_score("我养了一只叫笨笨的猫",
                                      filter =lambda x: x["page"] in [1,2,3,4,5,6,7,8,9,10])
for one in res:
    print(one)