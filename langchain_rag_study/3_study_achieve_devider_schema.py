'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 3_study_achieve_devider_schema.py
* @Time: 2025/9/9
* @All Rights Reserve By Brtc
'''
import os
from operator import itemgetter

import dotenv
import weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()
def format_qa_pair(question:str, answer:str)-> str:
    return f"Question: {question}\nAnswer: {answer}"

#1、定义分解子问题的prompt
decomposition_prompt = ChatPromptTemplate.from_template(
     "你是一个乐于助人的AI助理，可以针对一个输入问题生成多个相关的子问题。\n"
    "目标是将输入问题分解成一组可以独立回答的子问题或子任务。\n"
    "生成与以下问题相关的多个搜索查询：{question}\n"
    "并使用换行符进行分割，输出（3个子问题/子查询）:"
)
#2、构建分解问题链
decomposition_chain = (
    {"question":RunnablePassthrough()}
    |decomposition_prompt
    |ChatOpenAI(model="gpt-4o-mini", temperature=0)
    |StrOutputParser()
)
#3、构建向量数据库与检索器
client = weaviate.connect_to_weaviate_cloud(
    skip_init_checks=True,
    cluster_url=os.getenv("WAEVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = WeaviateVectorStore(client=client,
                         index_name="DataSetTest",
                         text_key="text",
                         embedding=embedding)
retriever = db.as_retriever(search_type = "mmr")
#4、执行提问获取子问题
question = "关于LLMOPS应用配置文档有哪些？"
all_question = decomposition_chain.invoke(question)
print(all_question)
sub_questions = all_question.split("\n")
#5、构建迭代问答链
prompt = ChatPromptTemplate.from_template(""" 
            这是你需要回答的问题：
            ---
            {question}
            ---
            这是所有可用的背景问题和答案对：
            ---
            {qa_pairs}
            ---
            这是与问题相关的额外背景信息：
            ---
            {context}
            ---
            使用上述背景信息和所有可用的背景问题和答案对来回答这个问题：
            {question}
            """)
chain =(
    {
        "context":itemgetter("question")|retriever,
        "question":itemgetter("question"),
        "qa_pairs":itemgetter("qa_pairs")
    }
    |prompt
    |ChatOpenAI(model="gpt-4o-mini",temperature=0)
    |StrOutputParser()
)
# 5、循环遍历所有子问题进行检索并获取答案
qa_pairs = ""
for sub_question in sub_questions:
    answer = chain.invoke({"question":sub_question,"qa_pairs":qa_pairs})
    qa_pairs += "\n----------\n" + format_qa_pair(sub_questions, answer)
    print(f"问题:{sub_question}")
    print(f"答案:{answer}")
print("============================")
client.close()