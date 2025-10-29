'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 19_study_conversation_chain.py
* @Time: 2025/9/3
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
chain = ConversationChain(llm=llm)
content = chain.invoke({"input":"你好我叫博小睿， 我喜欢篮球 唱跳  rap， 你喜欢什么运动"})
print(content)
content = chain.invoke({"input":"请根据上下文信息, 统计一下我喜欢什么运动？"})
print(content)