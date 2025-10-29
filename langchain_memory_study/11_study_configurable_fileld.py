'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 11_study_configurable_fileld.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
# 1、创建提示词模板
prompt = ChatPromptTemplate.from_template("请生成一个小于{x}随机数")
# 2、创建大语言模型
llm = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    temperature=ConfigurableField(id="llm_temperature", name = "大预言模型温度参数", description="用于调整大模型的温度")
)

# 3、构建链应用
chain = prompt | llm | StrOutputParser()
# 调用链
#content = chain.with_config(configurable={"llm_temperature":0}).invoke({"x":1000})
content = chain.invoke({"x":1000}, config={"configurable":{"llm_temperature":0}})
print(content)
