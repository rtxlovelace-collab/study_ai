'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 2_study_prompt_route.py
* @Time: 2025/9/10
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
dotenv.load_dotenv()
# 1、定义两份不同的prompt模板
physics_template = """ 
你是一位非常聪明的物理教程。
你擅长以简洁易懂的方式回答物理问题。
当你不知道问题的答案时，你会坦率承认自己不知道。
 
这是一个问题：
{query}
"""
math_template=""" 
你是一位非常优秀的数学家。你擅长回答数学问题。
你之所以如此优秀，是因为你能将复杂的问题分解成多个小步骤。
并且回答这些小步骤，然后将它们整合在一起回来更广泛的问题。
 
这是一个问题：
{query} 
"""
other_tempalte = "请回答用户的问题:{query}"
#2、创建嵌入文本模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
prompt_template = [physics_template, math_template, other_tempalte]
prompt_embeddings = embeddings.embed_documents(prompt_template)

"""
def prompt_router(input)->ChatPromptTemplate:
    #根据传递的query 计算返回不同的提示模板
    #1、计算传入的query嵌入向量
    query_embd = embeddings.embed_query(input["query"])
    #2、计算相似性
    tmp = cosine_similarity([query_embd], prompt_embeddings)
    print(tmp)
    similarity = tmp[0]
    most_similarity = prompt_template[similarity.argmax()]
    print("使用数学模板" if most_similarity == math_template else "使用物理模板")

    return ChatPromptTemplate.from_template(most_similarity)
"""
def prompt_router(input)->ChatPromptTemplate:
    #1、提示词模板
    question_class_prompt = ChatPromptTemplate.from_template("""
    ### 角色:
    你是一个优秀的问题分类专家，能够将人类问题进行整理并分类
    ### 规则:
    用户的问题分类总类为:数学问题、物理问题、其他问题
    如果，用户的问题为物理问题则回复0， 如果用户的问题为数学问题则回复1， 其他问题回复2
    ### 例子:
    Human:能介绍一下余弦公式吗？
    AI:1
    Human:黑洞是什么？
    AI:0 
    ### 限制
    1、严格按照规则内容进行回复
    2、请严格参考例子进行回复
    3、不要废话直接给出结果
    用户的问题是:{query}
    """)
    content = question_class_prompt|ChatOpenAI(model="gpt-4o-mini")|StrOutputParser()
    index = content.invoke(input["query"])
    print(prompt_template[int(index)])
    return ChatPromptTemplate.from_template(prompt_template[int(index)])

chain = (
    {"query":RunnablePassthrough()}
    |RunnableLambda(prompt_router)
    |ChatOpenAI(model="gpt-4o-mini")
    |StrOutputParser()
)

print(chain.invoke("牛顿第一定律是什么？"))
print("=================================")
print(chain.invoke("什么是洛伦磁力"))