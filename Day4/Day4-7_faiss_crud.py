import dotenv
from langchain_community.vectorstores import FAISS # 导入FAISS
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#10个文本
texts = (["笨笨是一只很喜欢睡觉的猫咪",
          "我喜欢在夜晚听音乐，这让我感到放松。",
          "猫咪在窗台上打盹，看起来非常可爱。",
          "学习新技能是每个人都应该追求的目标。",
          "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
          "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
          "我的手机突然关机了，让我有些焦虑。",
          "阅读是我每天都会做的事情，我觉得很充实。",
          "他们一起计划了一次周末的野餐，希望天气能好。",
          "我的狗喜欢追逐球，看起来非常开心。", ]
)
#FAISS 是一个开源的高效的向量搜索库，可以用于大规模向量搜索。
# 这里我们使用FAISS来存储和检索文本向量。
# 开始存入到faiss中
db = FAISS.from_texts(texts, embeddings,)
print(db.index_to_docstore_id)
#数据删除操作
print("删除前的总数：",db.index.ntotal)
db.delete([db.index_to_docstore_id[0]])
print("删除后的总数：",db.index.ntotal)
#数据保存到本地
db.save_local("./faiss_store/")
print("数据以保存到本地")
#数据从本地加载到内存
newdb = FAISS.load_local("./faiss_store/",
                         embeddings=embeddings,
                         allow_dangerous_deserialization=True)
print("数据加载成功")
docs=newdb.similarity_search_with_score("笨笨喜欢睡觉吗？")
for one in docs:
    print(one)
