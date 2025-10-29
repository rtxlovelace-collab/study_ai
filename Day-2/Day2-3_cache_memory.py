from operator import itemgetter

import dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory
dotenv.load_dotenv()
#创建提示词
prompt =ChatPromptTemplate.from_messages([
    ("system","你是基于OpenAI的聊天机器人，请按照用户问题回答问题"),
    MessagesPlaceholder("history"),
    ("human","{query}")

        ])
#构建大模型
llm = ChatOpenAI(model = "gpt-4o-mini")
#记忆
# memory = ConversationBufferWindowMemory(k=4,return_messages=True,input_key="query")
memory = ConversationTokenBufferMemory(return_messages=True,
                                       input_key="query",
                                       llm=llm,
                                       max_token_limit=10)
memory_variable = memory.load_memory_variables
#构建链
chain= (RunnablePassthrough.assign(history=RunnableLambda(memory_variable) | itemgetter("history"))|
        prompt|llm | StrOutputParser())

while True:
    query = input("Human：")
    if query == "q":
        print("Goodbye")
        exit(0)
    chain_input = {"query":query,"history":[]}
    resonse = chain.stream(chain_input)
    print("AI：",flush=True,end="")
    ai_output=""
    for chunk in resonse:
        print(chunk,end="",flush=True)
        chunk+=chunk
    memory.save_context(chain_input,{"output":ai_output})
    print("\n\n histoey:",memory.load_memory_variables({}))