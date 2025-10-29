'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 18_study_doc_chain.py
* @Time: 2025/9/3
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
dotenv.load_dotenv()
# 1、创建提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大的聊天机器人，能根据用户的上下文来回答问题\n\n, <context>{context}</context>"),
    ("human","{query}")
])
#2、创建大模型
llm = ChatOpenAI(model="gpt-4o-mini")
#3、创建链应用
chain = create_stuff_documents_chain(prompt=prompt, llm=llm)
#4、文档列表
documents = [
    Document(page_content="小明喜欢绿色,但不喜欢黄色"),
    Document(page_content="小王喜欢粉色,也有一点喜欢红色"),
    Document(page_content="小泽喜欢蓝色,但更喜欢青色"),
]
content = chain.invoke({"query":"请帮我统计一下大家都喜欢什么颜色", "context":documents})
print(content)