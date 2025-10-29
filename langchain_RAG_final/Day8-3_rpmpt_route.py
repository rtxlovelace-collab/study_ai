import dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
physics_template = """
            你是一位非常聪明的物理教授。
            你擅长以简洁易懂的方式回答物理问题。
                当你不知道问题的答案时，你会坦率承认自己不知道。

            这是一个问题：
            {query}"""
math_template = """
            你是一位非常聪明的数学教授。
            你擅长以简洁易懂的方式回答数学问题。
                当你不知道问题的答案时，你会坦率承认自己不知道。

            这是一个问题：
            {query}"""

#嵌入文本模型
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
prompt_template = [physics_template, math_template]
prompt_embedding =embeddings.embed_documents(prompt_template)
'''
def prompt_router(input)->ChatPromptTemplate:
    #根据传递的query计算返回不同的提示词模板
    #计算传入的query的embedding
    query_embedding = embeddings.embed_query(input["query"])
    #计算query和prompt的相似度
    tmp = cosine_similarity([query_embedding], prompt_embedding)
    print(tmp)
    similarity = tmp[0]
    
    
    
    '''
def prompt_router(input)->ChatPromptTemplate:
    question_class_prompt = ChatPromptTemplate.from_template("""
            ### 角色:
            你是一个优秀的问题分类专家，能够将人类问题进行整理并分类
            ### 规则:
            用户的问题分类总类为：数学问题、物理问题、其他问题
            如果，用户的问题为物理问题则回复 0, 如果用户的问题为数学问题则回复 1, 其他问题回复 2
            ### 例子:
            Human: 能介绍一下余弦公式吗？
            AI:1
            Human: 黑洞是什么？
            AI:0
            ###限制
            1、严格按照规则内容进行回复
            2、请严格参考例子进行回复
            3、不要废话直接给出结果
            用户的问题是:{query}
                    
            """)
    chain=question_class_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)|StrOutputParser()
    index = chain.invoke(input["query"])
    print(prompt_template[int(index)])
    return ChatPromptTemplate.from_template(prompt_template[int(index)])


chain = (
        {"query": RunnablePassthrough()}
        | RunnableLambda(prompt_router)
        | ChatOpenAI(model="gpt-4o-mini", temperature=0)
        | StrOutputParser()
)

print(chain.invoke("黑洞是什么?"))
print("======================")
print(chain.invoke("能介绍下余弦计算公式么？"))


