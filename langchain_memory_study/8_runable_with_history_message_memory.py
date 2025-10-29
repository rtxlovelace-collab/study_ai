'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 8_runable_with_history_message_memory.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import  RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
# 1、定义存储历史消息的字典
store = {}
#2、工厂函数，用于获取指定会话的历史消息
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = FileChatMessageHistory(f"chat_history_{session_id}.txt")
    return store[session_id]
#3、构建提示词 模板与大语言模型
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大的聊天机器人,请根据用户的需求回复问题"),
    MessagesPlaceholder("history"),
    ("human", "{query}")
])
llm = ChatOpenAI(model="gpt-4o-mini")
# 4、构建链
chain = prompt|llm|StrOutputParser()
# 5、包装链
with_massage_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)
while True:
    query = input("Human:>")
    if query == "exit":
        print("GoodBye!!!")
        exit(0)
    response = with_massage_chain.stream(
        {"query":query},
        config = {"configurable":{"session_id":"brtc_altman"}}
    )
    print("AI: ", end = "", flush=True)
    for chunck in response:
        print(chunck, end="", flush=True)
    print("")