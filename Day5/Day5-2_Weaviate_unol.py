import os
import dotenv
import weaviate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter
from weaviate.classes.config import Property, DataType  # 新增导入
dotenv.load_dotenv()
'''
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
'''

#连接至本地向量数据库
client = weaviate.connect_to_local(os.getenv("WEAVIATE_HOST"))
#定义嵌入式模型
embedding=OpenAIEmbeddings(model="text-embedding-3-small")

texts=["笨笨是一只很喜欢睡觉的猫咪",
          "我喜欢在夜晚听音乐，这让我感到放松。",
          "猫咪在窗台上打盹，看起来非常可爱。",
          "学习新技能是每个人都应该追求的目标。",
          "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
          "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
          "我的手机突然关机了，让我有些焦虑。",
          "阅读是我每天都会做的事情，我觉得很充实。",
          "他们一起计划了一次周末的野餐，希望天气能好。",
          "我的狗喜欢追逐球，看起来非常开心。", ]
metadatas : list=[
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
]
'''
在与向量数据库（如 Weaviate、Pinecone 等）或文档检索相关的场景中，metadatas（元数据）主要有以下作用：
1. 附加信息存储
metadatas 里的每个字典，是为对应 texts 中的文本提供额外的描述性信息。
比如示例中，{"page": 1} 表示 texts 里第一个文本对应的 “页码是 1”；{"page": 5, "account_id": 1} 则补充了 “页码是 5” 且 “账号 ID 为 1” 的信息。
这些信息不是文本内容本身，但能从不同维度描述文本。
2. 过滤与检索优化
在进行向量相似性检索时，除了基于文本语义匹配，还可以结合元数据进行过滤。
例如，若想只检索 “页码小于 5” 的文本，就可以利用 metadatas 里的 page 字段来设置过滤条件，缩小检索范围，提高检索的精准度和效率。
3. 上下文与溯源
元数据能帮助理解文本的上下文或来源。像 “页码” 可以对应文本在某本书、某份文档中的位置，“account_id” 可关联到生成或拥有该文本的账户等，
方便后续对文本进行管理、溯源或关联分析。
简单来说，texts 是核心内容，metadatas 则是给这些内容 “贴标签、加说明”，让文本在存储、检索、管理时能携带更多背景信息，更灵活地满足各种场景需求。
'''
# 如果集合不存在则创建，并定义元数据属性
if not client.collections.exists("DatasetTest"):
    client.collections.create(
        name="DatasetTest",
        properties=[
            Property(name="text", data_type=DataType.TEXT),  # 存储文本内容
            Property(name="page", data_type=DataType.INT),   # 定义page属性为整数类型
            Property(name="account_id", data_type=DataType.INT)  # 定义account_id属性
        ]
    )
db = WeaviateVectorStore(client=client,index_name="DatasetTest",
                          text_key="text", embedding=embedding,)

# 先添加数据（如果还没添加过）
ids = db.add_texts(texts=texts, metadatas=metadatas)
print(ids)


#过滤搜索
filters = Filter.by_property("page").greater_than(5)

text= db.similarity_search_with_score("笨笨", filters=filters)
for one in text:
    print(one)
client.close()
