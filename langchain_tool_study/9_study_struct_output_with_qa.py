'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 9_study_struct_output_with_qa.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
dotenv.load_dotenv()
class QAExtra(BaseModel):
    """一个问答键值对工具, 传递对应的假设性问题+答案"""
    question:str = Field(description = "假设性问题")
    answer:str = Field(description="假设性问题的对应的答案")
llm = ChatOpenAI(model="gpt-4o-mini")
struct_llm = llm.with_structured_output(QAExtra, method="json_mode")
prompt = ChatPromptTemplate.from_messages([
    ("system", "请从用户传递的query中提取出假设性的问题+答案，响应格式为JSON， 并携带'question'和'answer'两个字段。"),
    ("human","{query}")
])
chain = {"query":RunnablePassthrough()}|prompt|struct_llm
print(chain.invoke("我叫博小睿，喜欢篮球，游泳唱跳和rap"))