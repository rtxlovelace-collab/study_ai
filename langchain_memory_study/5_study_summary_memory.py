'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_study_summary_memory.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_study_cache_memory.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
from operator import itemgetter

import dotenv
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, \
    ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
# 1、创建提示词
prompt = ChatPromptTemplate.from_messages([
    ("system","你是OpenAI研发的聊天机器人,请按照实际情况回答用户的问题"),
    MessagesPlaceholder("history"),
    ("human", "{query}")
])
# 记忆

# 构建大模型
llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationSummaryBufferMemory(return_messages=True, input_key="query", llm = llm,max_token_limit = 300)


# 构建链应用
chain = (RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))
         |prompt|llm|StrOutputParser())
while True:
    query = input("Human:")
    if query == "exit":
        print("GoodBye!!!")
        exit(0)
    chain_input = {"query":query, "history":[]}
    response = chain.stream(chain_input)
    print("AI:", flush=True, end="")
    ai_output = ""
    for chunk in response:
        print(chunk, end="", flush=True)
        chunk += chunk
    memory.save_context(chain_input,{"output":ai_output})
    print("\n\n history: ", memory.load_memory_variables({}))


