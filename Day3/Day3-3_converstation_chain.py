import dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain_core import documents
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
chain = ConversationChain(llm=llm)
'''ConversationChain 是 LangChain 框架中用于构建简单对话交互的核心组件之一，
核心作用是让 AI 能够基于「历史对话上下文」与用户进行连贯交互，而非每次回复都 “断联” 式仅依赖当前输入。'''
content1=chain.invoke({"input":"你好我是Jerry，喜欢的游戏比如战地风云2042，我的世界，刺客信条之大革命，无人深空"})
print("第一段",content1)
content= chain.invoke({"input":"说说我喜欢什么游戏呢？"})
print("第二段",content)