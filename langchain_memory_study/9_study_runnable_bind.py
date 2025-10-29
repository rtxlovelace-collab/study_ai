'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 9_study_runnable_bind.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
prompt = ChatPromptTemplate.from_messages([
    ("system", ""),
    ("human", "{query}")
])
llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt|llm.bind(model="gpt-4")|StrOutputParser()
content = chain.invoke({"query":"你是Gpt几呀？"})
print(content)