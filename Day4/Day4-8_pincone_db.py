import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore#导入pinecone库

dotenv.load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



#Pincoen是一个开源的云向量库，它可以将文本数据转换为向量，并将向量存储在Pinecone云向量库中。
# 它用于连接Pinecone集群并进行向量检索、相似度搜索等操作。

# 首先，我们需要创建一个Pincoen客户端，并连接到Pinecone向量库。
db=PineconeVectorStore(index_name="text",embedding=embeddings,
                       namespace="study")
'''
                        index_name为向量库的索引名称
                        embedding为OpenAI的文本嵌入模型，namespace为命名空间名称。
'''

# 这里向Pincoen数据库中添加文本数据，并为每个文本指定一个元数据字典。
db.add_texts(["笨笨是一只很喜欢睡觉的猫咪",
          "我喜欢在夜晚听音乐，这让我感到放松。",
          "猫咪在窗台上打盹，看起来非常可爱。",
          "学习新技能是每个人都应该追求的目标。",
          "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
          "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
          "我的手机突然关机了，让我有些焦虑。",
          "阅读是我每天都会做的事情，我觉得很充实。",
          "他们一起计划了一次周末的野餐，希望天气能好。",
          "我的狗喜欢追逐球，看起来非常开心。", ]
, [
    {"page": 1},
    {"page": 2},
    {"page": 3},
    {"page": 4},
    {"page": 5,"account_id": 1},
    {"page": 6},
    {"page": 7},
    {"page": 8},
    {"page": 9},
    {"page": 10},
],namespace="study")
# 然后，我们可以检索数据库中与给定文本最相似的文本。
print(db.similarity_search_with_relevance_scores
      ("我养了一只叫笨笨的猫",namespace="study"))
