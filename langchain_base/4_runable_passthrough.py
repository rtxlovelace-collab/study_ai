'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_runable_passthrough.py
* @Time: 2025/9/1
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
# 1、编排 prompt
prompt = ChatPromptTemplate.from_template("{query}")
# 2、创建大模型
llm  = ChatOpenAI(model="gpt-4o-mini")
# 3、创建链
chain = {"query": RunnablePassthrough()}|prompt|llm|StrOutputParser()
# 4、调用链
print(chain.invoke("你好你叫什么名字？"))