import os
from operator import index

import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import MetaField, META_FIELD_TYPE_STRING,ConnectionParams
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from torch.fx.experimental.unification.dispatch import namespace

dotenv.load_dotenv()
#创建OpenAI的embedding
embedding=OpenAIEmbeddings(model='text-embedding-3-small')
#连接腾讯云数据库
db= TencentVectorDB(
    embedding=embedding,
    connection_params=ConnectionParams(
    url=os.environ.get('TC_VECTOR_DB_URL'),
    username=os.environ.get("TC_VECTOR_DB_USERNAME"),
    key=os.environ.get("TC_VECTOR_DB_KEY"),
    timeout=int(os.environ.get('TC_VECTOR_DB_TIMEOUT'))
                ),
    database_name=os.environ.get('TC_VECTOR_DB_DATABASE'),
    collection_name="demo",
    meta_fields=[
        MetaField(name="text",data_type=META_FIELD_TYPE_STRING,
                  index=False,
                  description="text description")])


#添加文本数据
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
#添加元数据
metadata=[{"text":text,"page":index} for index,text in enumerate(texts)]
#添加文本数据到数据库
ids=db.add_texts(texts,metadata)
print("添加文件",ids)