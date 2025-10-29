'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 6_presention_memory.py
* @Time: 2025/9/1
* @All Rights Reserve By Brtc
'''
from operator import itemgetter

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
def rertriever_from_qa(query:str):
    print(f"我正在向量数据库里面检索用的问题:{query}")
    return "Human：你好我叫博小睿，喜欢篮球唱跳rap， 请问你叫什么了？AI:请问您有什么问题需要帮助吗"
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
#3、编排链
chain = RunnableParallel({
    "context":lambda x:rertriever_from_qa(x["query"]),
    "query":itemgetter("query")
})|prompt|llm|parser
content = chain.invoke({"query":"请问我叫什么？"})
print(content)