'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_fewshot_prompt_template.py
* @Time: 2025/9/9
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
# 1、构建示例模板样例
examples=[
    {"question":"帮我计算一下2+2等于多少？", "answer":"4"},
    {"question":"帮我计算一下2+3等于多少？", "answer":"5"},
    {"question":"帮我计算一下2*15等于多少？", "answer":"30"}
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human","{question}"),
    ("ai", "{answer}")
])
# 2、构建少量示例提示词模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples=examples,
)
print("少量提示词示例模板:", few_shot_prompt.format())
# 3、构建最终的提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个可以计算复杂问题的聊天机器人,请直接返回计算结果，参考示例模板"),
    few_shot_prompt,
    ("human", "{question}")
])
llm = ChatOpenAI(model="gpt-4o-mini")
chain = prompt|llm|StrOutputParser()
print(chain.invoke("帮我计算一下14*15等于多少"))