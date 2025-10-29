'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 1_runable_parallel_use.py
* @Time: 2025/9/1
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
#1、编排prompt
joke_prompt = ChatPromptTemplate.from_template("请讲一个关于{subject}的冷笑话")
poem_prompt = ChatPromptTemplate.from_template("请写一首关于{subject}的诗歌")
#2、创建大语言模型
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
#3、编排链
joke_chain = joke_prompt|llm|parser
poem_chain = poem_prompt|llm|parser
#4、通过 runableparallel 并行链
map_chain = RunnableParallel(joke = joke_chain, poem = poem_chain)
#5、运行并获取结果
res = map_chain.invoke({"subject":"程序员"})
print(res)