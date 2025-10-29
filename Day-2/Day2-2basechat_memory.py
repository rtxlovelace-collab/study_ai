
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_memory.chat_memory import BaseChatMemory
import dotenv
#作业：把ai使用BaseChatMemory来记忆，并实现一个简单的聊天机器人
dotenv.load_dotenv()
parser = StrOutputParser()
# class ConversationTokenBufferMemory(BaseChatMemory):


#编排prompt
prompt = ChatPromptTemplate.from_template("讲一个关于{job}的故事")
#构建大语言模型
llm = ChatOpenAI(model='gpt-4o-mini')
#编排链

chain_story = prompt | llm | parser


memory = BaseChatMemory(input_key="query",output_key="output",return_message=True)
memory_var = memory.load_memory_variavles({})






#获取记忆添加到提示词中
content = chain_story.invoke({"query": "讲一个关于程序员的故事"},{"chat_history":""})