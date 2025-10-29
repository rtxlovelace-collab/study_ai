'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 3_study_self_retriever.py
* @Time: 2025/9/10
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers import SelfQueryRetriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

dotenv.load_dotenv()
# 1、构建文档
# 1.构建文档列表并上传到数据库
documents = [
    Document(
        page_content="肖申克的救赎",
        metadata={"year": 1994, "rating": 9.7, "director": "弗兰克·德拉邦特"},
    ),
    Document(
        page_content="霸王别姬",
        metadata={"year": 1993, "rating": 9.6, "director": "陈凯歌"},
    ),
    Document(
        page_content="阿甘正传",
        metadata={"year": 1994, "rating": 9.5, "director": "罗伯特·泽米吉斯"},
    ),
    Document(
        page_content="泰坦尼克号",
        metadat={"year": 1997, "rating": 9.5, "director": "詹姆斯·卡梅隆"},
    ),
    Document(
        page_content="千与千寻",
        metadat={"year": 2001, "rating": 9.4, "director": "宫崎骏"},
    ),
    Document(
        page_content="星际穿越",
        metadat={"year": 2014, "rating": 9.4, "director": "克里斯托弗·诺兰"},
    ),
    Document(
        page_content="忠犬八公的故事",
        metadat={"year": 2009, "rating": 9.4, "director": "莱塞·霍尔斯道姆"},
    ),
    Document(
        page_content="三傻大闹宝莱坞",
        metadat={"year": 2009, "rating": 9.2, "director": "拉库马·希拉尼"},
    ),
    Document(
        page_content="疯狂动物城",
        metadat={"year": 2016, "rating": 9.2, "director": "拜伦·霍华德"},
    ),
    Document(
        page_content="无间道",
        metadat={"year": 2002, "rating": 9.3, "director": "刘伟强"},
    ),
]

# 元素据创建
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="电影的发布年份",
        type="integer",
    ),
    AttributeInfo(
        name="rating",
        description="电影的评分",
        type="float",
    ),
    AttributeInfo(
        name="director",
        description="电影的导演",
        type="string",
    ),
]

db = PineconeVectorStore(index_name="for-study",
                         embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
                         namespace="selfretriever",
                         text_key="text")

#db.add_documents(documents)
test = db.similarity_search("查找一下评分高于9.0分的电影", filter={"rating":{"$gt":9.0}})
for one in test:
    print(one)

# 3、创建查询检索
self_query_retriever = SelfQueryRetriever.from_llm(
    llm = ChatOpenAI(model="gpt-4o-mini"),
    vectorstore=db,
    document_contents="电影的名字",
    metadata_field_info=metadata_field_info,
    enable_limit=True
)

#4、检索
results = self_query_retriever.invoke("查找一下评分高于9.0分的电影", config={"k":10})

for  one in results:
    print("=============================================")
    print(one)