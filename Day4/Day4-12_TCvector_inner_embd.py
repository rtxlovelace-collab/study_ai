import os
from operator import index

import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import MetaField, META_FIELD_TYPE_STRING,ConnectionParams
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from torch.fx.experimental.unification.dispatch import namespace

dotenv.load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-small')

db= TencentVectorDB(
    embedding=None,
    connection_params=ConnectionParams(
    url=os.environ.get('TC_VECTOR_DB_URL'),
    username=os.environ.get("TC_VECTOR_DB_USERNAME"),
    key=os.environ.get("TC_VECTOR_DB_KEY"),
    timeout=int(os.environ.get('TC_VECTOR_DB_TIMEOUT'))
                ),
    database_name=os.environ.get('TC_VECTOR_DB_DATABASE'),
    collection_name="demo1",
    meta_fields=[
        MetaField(name="text",data_type=META_FIELD_TYPE_STRING,
                  index=False,
                  description="text description"),
        MetaField(name="page", data_type=META_FIELD_TYPE_STRING,
          index=False,
          description="page")]
)
'''
embedding=None：表示不使用内置的向量嵌入模型，可能会在后续操作中单独处理向量生成。
connection_params：配置数据库连接参数
通过环境变量获取数据库的 URL、用户名、密钥和超时时间
这种方式避免了硬编码敏感信息，提高安全性
database_name：指定要连接的数据库名称，同样从环境变量获取
collection_name="demo1"：指定操作的数据集合（类似表）名称为 "demo1"
meta_fields：定义元数据字段
包含 "text" 字段：字符串类型，不建立索引，用于存储文本描述
包含 "page" 字段：字符串类型，不建立索引，用于存储页面信息
'''

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
# 向数据库中添加文本数据
metadata=[{"text":text,"page":index} for index,text in enumerate(texts)]
#
ids=db.add_texts(texts)
print("添加文件id",ids)
print(db.similarity_search_with_score("我养了一只猫，叫笨笨"))