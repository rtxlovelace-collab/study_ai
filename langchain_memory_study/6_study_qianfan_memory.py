'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 6_study_qianfan_memory.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_study_cache_memory.py
* @Time: 2025/6/23
* @All Rights Reserve By Brtc
'''
from operator import itemgetter

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, ConversationTokenBufferMemory
from langchain_community.chat_models.baidu_qianfan_endpoint import  QianfanChatEndpoint


dotenv.load_dotenv()
#1、创建提示词模板& 记忆
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是OpenAI 开发的机器人，请根据对应的上下文来回答用户的问题。"),
    MessagesPlaceholder("history"),
    ("human", "{query}"),
])

# 记忆
"""
memory = ConversationBufferWindowMemory(k=2,
                                        return_messages=True,
                                        input_key="query")
"""
memory = ConversationSummaryBufferMemory(
                                        max_token_limit=300,
                                        return_messages=True,
                                        input_key="query",
                                         llm=ChatOpenAI(model='gpt-3.5-turbo-16k'))

memory_variable = memory.load_memory_variables({})
#2、创建大语言模型

llm = QianfanChatEndpoint(api_key="ALTAK8yWpDcL811FmasGRHh1m7", secret_key = "725e6f236e5947f69c0ab8a0bbfc14a1")

#3、构建链应用
chain = RunnablePassthrough.assign(history= RunnableLambda(memory.load_memory_variables) | itemgetter("history")) | prompt | llm |StrOutputParser()

#4、构建对话命令行
while True:
    query = input("Human:")

    if query == "exit":
        exit(-1)

    chain_input = {"query": query, "history": []}

    response = chain.stream(chain_input)
    print("AI:", flush=True, end="")
    ai_output=""
    for chunk in response:
        print(chunk, flush=True, end="")
        ai_output += chunk
    memory.save_context(chain_input, {"output": ai_output})
    print("")
    print("history:", memory.load_memory_variables({}))