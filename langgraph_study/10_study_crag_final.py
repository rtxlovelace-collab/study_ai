'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 10_study_crag_final.py
* @Time: 2025/9/16
* @All Rights Reserve By Brtc
'''
import os
from typing import Any

import dotenv
import weaviate
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()

class GradeDocument(BaseModel):
    '''文档质量评估数据模型'''
    binary_score:str = Field(description="文档与问题是否关联,请回答yes or no")

class GoogleSerperArgsSchema(BaseModel):
    query:str = Field(description="执行谷歌搜索的查询语句")

class GraphState(TypedDict):
    '''图结构对应的数据状态'''
    question:str # 原始问题
    generation:str # 大语言模型生成内容
    web_search:str # 网络搜索内容
    documents:list[Document]#检索内容

def format_docs(docs:list[Document])-> str:
    '''拼接文档列表成为一个字符串'''
    return "\r\n".join([doc.page_content for doc in docs])

# 创建大预言模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 创建 检索器
client = weaviate.connect_to_weaviate_cloud(
    skip_init_checks=True,
    cluster_url=os.getenv("WAEVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = WeaviateVectorStore(client=client,
                         index_name="DataSetTest",
                         text_key="text",
                         embedding=embedding)
retriever = db.as_retriever(search_type="mmr")
# 构建评估器
system = """ 
你是一名评估检索到的文档与用户问题相关性的评估员。
如果文档包含与问题相关的关键字或语义，请将其评级为相关。
给出一个是否相关得分为yes或者no，以表明文档是否与问题相关。
"""
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "检索文档:\n\n {document}\n\n 用户问题:{question}")
])
retriever_grade = grade_prompt|llm.with_structured_output(GradeDocument)
# rag检索增强生成
template = """ 
你是一个问答任务的助理。使用以下检索到的上下文来回答问题。如果不知道就说不知道，不要胡编乱造，并保持答案简洁。
问题: {question}
上下文: {context}
答案: 
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = prompt|llm.bind(temperature=0)|StrOutputParser()
#网络问题重写
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个将输入问题转换为优化的更好版本的问题重写器并用于网络搜索。请查看输入并尝试推理潜在的语义意图/含义。"),
    ("human",  "这里是初始化问题:\n\n{question}\n\n请尝试提出一个改进问题。")
])
question_rewriter = rewrite_prompt|llm.bind(temperature=0)|StrOutputParser()
# 网络检索工具
google_serper = GoogleSerperRun(
    name = "google_serper",
    description = "一个低成本的谷歌搜索API 工具, 当你需要回答关键实事的 时候可以调用该工具",
    api_wrapper=GoogleSerperAPIWrapper(),
    args_schema=GoogleSerperArgsSchema
)
# 构建图的关键工具
def retriver(state:GraphState)->Any:
    """检索节点，根据用户的问题检索向量数据库"""
    print("-----------------检索节点------------------------")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents":documents, "question":question}

def generate(state:GraphState)->Any:
    """生成节点，根据原始问题 + 检索上下文内容调用LLM生成的内容"""
    print("-----------------生成内容节点------------------------")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({
        "context":format_docs(documents),
        "question":question
    })
    return {"question":question,"generation":generation,"documents":documents}

def grade_documents(state:GraphState)->Any:
    """文档与原始问题关联性评估节点"""
    print("-----------------关联性评估节点------------------------")
    question = state["question"]
    documents = state["documents"]

    filter_docs = []
    web_search="no"
    for doc in documents:
        score:GradeDocument = retriever_grade.invoke({
            "question":question,
            "document":doc.page_content
        })
        grade = score.binary_score
        if grade.lower() == "yes":
            print("$$$$$$文档存在关联$$$$$$$")
            filter_docs.append(doc)
        else:
            print("$$$$$$文档不存在关联$$$$$$$")
            web_search="yes"
            continue
    return {**state, "documents":filter_docs, "web_search":web_search}

def web_search(state:GraphState)->Any:
    """重写转化节点"""
    print("-----------------网络重写转换节点------------------------")
    question = state["question"]
    documents = state["documents"]
    search_content = google_serper.invoke({"query":question})
    documents.append(Document(page_content=search_content))
    return {**state, "documents":documents}

def transform_query(state:GraphState)->Any:
    """重写/转化查询节点"""
    print("-----------------问题转换节点------------------------")
    question = state["question"]
    better_question = question_rewriter.invoke({"question":question})
    return {**state, "question":better_question}

def decide_to_generate(state:GraphState)->Any:
    """决定生成还是搜索节点"""
    print("-----------------路由选择节点------------------------")
    web_search = state["web_search"]
    if web_search == "yes":
        print("$$$$$$执行检索$$$$$$$$$")
        return "transform_query"
    else:
        print("$$$$$$执行最终生成$$$$$$$$$")
        return "generate"

# 构建工作流程
work_flow = StateGraph(GraphState)

# 定义节点
work_flow.add_node("retriver", retriver)
work_flow.add_node("grade_documents", grade_documents)
work_flow.add_node("web_search", web_search)
work_flow.add_node("generate", generate)
work_flow.add_node("transform_query", transform_query)
# 定义边
work_flow.set_entry_point("retriver")
work_flow.add_edge("retriver", "grade_documents")
work_flow.add_conditional_edges("grade_documents", decide_to_generate)
work_flow.add_edge("transform_query", "web_search")
work_flow.add_edge("web_search", "generate")
work_flow.set_finish_point("generate")

app = work_flow.compile()

print(app.invoke({"question":"能介绍一下什么是LLMOPS吗？"}))
client.close()