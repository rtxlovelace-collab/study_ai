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
    embedding=embedding,
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
                  description="text description")])


'''TCVectorDB 也支持为相似性搜索添加过滤器，表达式的格式为 <field_name><operator><value>，多个表达式之间支持 and（与）、or（或）、not（非）关系。其中：
1. <field_name>：表示要过滤的字段名。
2. <operator>：要使用的运算符。
  - string ：匹配单个字符串值（=）、排除单个字符串值（!=）、匹配任意一个字符串值（in）、排除所有字符串值（not in）。其对应的 Value 必须使用英文双引号括起来。
  - uint64：大于（>）、大于等于（>=）、等于（=）、小于（<）、小于等于（<=）、不等于（!=）。
3. <value>：表示要匹配的值。'''
#例如
ids=db.similarity_search_with_score("我养了一只叫笨笨的猫",expr="page>5")
for one in ids:
    print(one)
'''db 是一个已经初始化的向量数据库客户端实例
similarity_search_with_score 是执行相似性搜索的方法，会返回相似的结果及其相似度分数
第一个参数 "我养了一只叫笨笨的猫" 是搜索的查询文本（会被转换为向量进行匹配）
expr="page>5" 是附加的过滤条件，限制只返回满足 "page 字段大于 5" 的结果


'''