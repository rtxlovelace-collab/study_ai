'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 15_study_llm_with_roll_back.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
# 1、构建链
prompt = ChatPromptTemplate.from_template("{query}")
llm = ChatOpenAI(model="gpt-999").with_fallbacks([ChatOpenAI(model="gpt-777"),ChatOpenAI(model="gpt-4o-mini")])

# 2、构建链
chain= prompt|llm|StrOutputParser()

# 3、调用并输出结果
content = chain.invoke({"query":"你好,你是？"})
print(content)