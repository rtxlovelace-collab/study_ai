
import os
import dotenv
import weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.classes.query import Filter
from weaviate.classes.config import Property, DataType  # 新增导入
dotenv.load_dotenv()
#创建一个历史对话列表存入历史对话
chat_histories=[]
# 连接至本地向量数据库
client = weaviate.connect_to_local(os.getenv("WEAVIATE_HOST"))
# 定义嵌入式模型
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# 如果集合不存在则创建，
if not client.collections.exists("DatasetTest"):
    client.collections.create(
        name="DatasetTest",
        properties=[
            Property(name="text", data_type=DataType.TEXT),  # 存储文本内容
            Property(name="account_id", data_type=DataType.INT)  # 定义account_id属性
        ]
    )
#WeaviateVectorStore可以将文本转化为向量数据
db = WeaviateVectorStore(client=client, index_name="DatasetTest",
                             text_key="text", embedding=embedding )

def weaviate_offlien(texts, db):
    #这是存入Weaviate数据库的函数方法
    # 确保输入格式正确（texts应为字符串列表）
    if not isinstance(texts, list) :
        raise ValueError("texts必须是列表类型")

    return db.add_texts(texts=texts)

def GetHistoricalConversations(chat_histories, human_query, ai_content):
    #这是获取历史对话并存入历史对话列表的函数方法
    chat_histories.append({"human":human_query,"ai": ai_content})
    return chat_histories

if __name__ == '__main__':
    while True:
        human_query = input("Human Query: ")
        if human_query == "q":
            print("Goodbye")
            client.close()
            exit(0)
        # 对历史对话用函数Filter（过滤器）按属性 by_property("account_id", 0)进行检索，附加条件是greater_than(0)
        filters = Filter.by_property("account_id", 0).greater_than(0)
        # 返回结果的处理，提取检索到的文本内容
        # similarity_search_with_score 是向量数据库 / 向量检索工具中常用的核心函数，
        # 核心作用是 “在向量集合中找到与目标向量最相似的结果，并返回相似性得分”
        retrieved_docs = db.similarity_search_with_score(human_query, filters=filters, k=10)  # 限制返回结果为10条
        # 提取向量数据库中文本内容并筛选出分数大于0.5的文本内容，rieved_texts就是历史对话,
        rieved_texts = [doc.page_content for doc, score in retrieved_docs if score > 0.5]
        # rieved_texts=[]
        # for doc,score in retrieved_docs:
        #     if score > 0.5:
        #         rieved_texts.append(doc.page_content)
        #         print(score)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是基于OpenAI开发的聊天机器人，请根据历史对话回答问题"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{query}")
        ])
        # 构造符合格式的历史消息
        llm = ChatOpenAI(model="gpt-4o-mini")
        # 定义lecel链条
        chain = prompt | llm | StrOutputParser()
        # 调用链时传入正确参数
        ai_output = chain.invoke({
            "history": rieved_texts,
            "query": human_query,
        })

        print("AI：", ai_output)
        # 保存本轮对话到历史
        # texts= [f"Human:{human_query} \nAI:{ai_output}"]
        chat_histories= GetHistoricalConversations(human_query,ai_output)
        # 将历史对话保存至数据库
        ids = weaviate_offlien([f"Human:{human_query} \nAI:{ai_output}"],db)  # 只存新增
        # ids = db.add_texts(texts=texts, metadatas=metadatas[-1:])  # 存全量
        print(ids)
'''
similarity_search_with_score

similarity_search_with_score 是向量数据库 / 向量检索工具中常用的核心函数，核心作用是 “在向量集合中找到与目标向量最相似的结果，并返回相似性得分”，本质是为了解决 “非结构化数据（如文本、图片、音频）的相似性匹配” 问题。
1. 核心功能拆解
“similarity_search”（相似性检索）：
先将目标数据（如一段文本、一张图片）转换成高维向量（通过模型如 BERT、ResNet 等），再在数据库已存储的 “向量库” 中，通过向量距离算法（如欧氏距离、余弦相似度），筛选出与目标向量 “距离最近” 的 N 个向量（即最相似的 N 条数据）。
“with_score”（附带得分）：
检索结果不仅包含 “最相似的数据本身”，还会返回相似性量化得分—— 得分的含义由所用的距离算法决定：
若用 “余弦相似度”：得分越接近 1，相似度越高；越接近 0，相似度越低。
若用 “欧氏距离”：得分（距离值）越小，相似度越高；值越大，相似度越低。
2. 典型使用场景
主要用于处理 “无法用传统关键词匹配解决” 的相似性需求，例如：
文本领域：文档相似性推荐（如 “你可能还喜欢的文章”）、语义搜索（输入 “如何养宠物”，能匹配到 “宠物喂养指南” 而非仅含 “养宠物” 关键词的文本）。
多媒体领域：图片相似检索（上传一张风景照，找到图库中同款场景的图片）、音频相似匹配（识别相似的背景音乐）。
AI 应用：大语言模型（LLM）的 “检索增强生成（RAG）”—— 先通过该函数从知识库中找到与用户问题相似的上下文，再让 LLM 基于这些上下文生成准确回答。
3. 关键参数（通用逻辑）
不同工具（如 Pinecone、FAISS、LangChain 中的实现）参数略有差异，但核心参数一致：
参数	作用说明	示例值
query_vector	目标向量（需与向量库中向量维度一致）	[0.12, 0.34,...]（512 维向量）
k	要返回的 “最相似结果数量”	5（返回 Top5 相似结果）
distance_metric	可选，指定计算相似性的算法（默认多为余弦相似度）	"cosine" / "euclidean"
4. 与 “similarity_search” 的区别
similarity_search：仅返回 “最相似的 N 条数据”，不提供量化得分，无法判断 “相似程度到底有多高”。
similarity_search_with_score：多返回一列得分，能帮助用户进一步筛选（例如只保留得分≥0.8 的高相似结果），或向用户展示 “匹配可信度”。
简单说，这个函数是 “相似性检索的增强版”—— 不仅告诉你 “哪几个最像”，还告诉你 “像到什么程度”，是向量检索落地场景中（如 RAG、推荐系统）的高频工具。
similarity_search_with_score函数的应用案例
除了similarity_search_with_score，还有哪些类似的函数？
如何选择合适的向量距离算法与similarity_search_with_score配合使用？
'''
ChatMenmoryHistory=[]