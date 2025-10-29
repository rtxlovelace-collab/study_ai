'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 2_study_runnable_retriever.py
* @Time: 2025/9/1
* @All Rights Reserve By Brtc
'''
from idlelib import query
from operator import itemgetter

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
def rertriever_from_qa(query:str):
    print(f"我正在向量数据库里面检索用的问题:{query}")
    return f"我的名字叫博小睿， 喜欢篮球、rap、和唱跳"
#1、构建提示词
prompt = ChatPromptTemplate.from_template(""" 
请根据用户的问题回答,可以参考对应上下文进行回答:
<context>
{context}
</context>
""")
#2、构建 大模型
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
#编排链
chain = RunnableParallel({
    "context":lambda x:rertriever_from_qa(x["query"]),
    "query":itemgetter("query")
})|prompt|llm|parser
content = chain.invoke({"query":"你好我是谁？"})
print(content)