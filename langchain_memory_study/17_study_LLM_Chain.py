'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 17_study_LLM_Chain.py
* @Time: 2025/9/3
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
prompt = ChatPromptTemplate.from_template("请将一个{subject}主题的笑话")
llm = ChatOpenAI(model="gpt-4o-mini")

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke("程序员"))