'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 13_study_mult_mode_config.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
# 1、创建提示词模板&定义大语言模型
prompt = ChatPromptTemplate.from_template("{query}")
llm = ChatOpenAI(model="gpt-3.5-turbo").configurable_alternatives(
    ConfigurableField(id = "llm"),
    gpt4 = ChatOpenAI(model="gpt-4"),
    gpt4mini = ChatOpenAI(model="gpt-4o-mini")
)
#2、构建链应用
chain = prompt|llm|StrOutputParser()
# 3、调用链
content = chain.invoke({"query":"你是Gpt几了？"},config={"configurable":{"llm":"gpt4"}})
print(content)

