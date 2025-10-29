'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 7_study_entity_memory.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import  ConversationChain

dotenv.load_dotenv()

# 构建大模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#编排链
chain = ConversationChain(
    llm = llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm = llm)
)

print(chain.invoke({"input":"你好我叫博小睿， 最近在学习langchain"}))
print(chain.invoke({"input":"我最喜欢编程语言是Python"}))
print(chain.invoke({"input":"我住在长沙"}))

# 查询实体对话
res = chain.memory.entity_store.store
print(res)