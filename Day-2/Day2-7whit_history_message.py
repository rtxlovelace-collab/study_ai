import dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables.history import Runnable, RunnableWithMessageHistory

dotenv.load_dotenv()

#定义储存历史de ditc
store={}
#工厂函数，用于获取指定会话的历史消息
def get_session_history(session_id:str) ->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=FileChatMessageHistory(f"chat_history_{session_id}.txt")
    return store[session_id]

#构建提示词 模板 大模型
prompt=ChatPromptTemplate.from_messages([
    ("system","你是一个强大的聊天机器人，请根据用户的问题回答"),
    MessagesPlaceholder("history"),
    ("human","{query}"),
])
llm= ChatOpenAI(model="gpt-4o-mini")
#构建会话链
chain=prompt|llm|StrOutputParser()
#包装链
with_massage_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)
#运行
while True:
    query = input("Human：")
    if query == "q":
        print("Goodbye")
        exit(0)

    response = with_massage_chain.stream(
        {"query":query},
        config={"configurable":{"session_id":"brtc_altman"}}
        )
    print("AI：",end="",flush=True)
    for chunk in response:
        print(chunk,end="",flush=True)
    print("")



