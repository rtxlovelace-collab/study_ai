import dotenv
import langchain_core
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableParallel

dotenv.load_dotenv()
parser = StrOutputParser()
#编排prompt
prompt1 = ChatPromptTemplate.from_template("说一个关于{job}的笑话")
prompt2 = ChatPromptTemplate.from_template("讲一个关于{job}的故事")
#构建大语言模型
llm = ChatOpenAI(model='gpt-4o-mini')
#编排链
chain_joke = prompt1 | llm | parser
chain_story = prompt2 | llm | parser
#通过runabeparallel并行链
map_chain = RunnableParallel(joke=chain_joke, story=chain_story)
#运行链
res=map_chain.invoke({"job":"程序员"})
print(res)