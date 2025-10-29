import dotenv
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
dotenv.load_dotenv()

#大模型
llm=ChatOpenAI(model="gpt-4o-mini",temperature=0.1)
#编排链
chain= ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm = llm),

)

print(chain.invoke({"input": "你好我叫Jerry，最近因为学习ai开发而放弃玩心爱的游戏"}))

#查询实体对话
res=chain.memory.entity_store.store
print(res)