'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_step_back_retriever.py
* @Time: 2025/9/9
* @All Rights Reserve By Brtc
'''
import os
from typing import List

import dotenv
import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()
class StepBackRetriever(BaseRetriever):
    """回答回退检索器"""
    retriever:BaseRetriever
    llm:BaseLanguageModel
    def _get_relevant_documents( self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """根据传递的query执行问题回退并检索"""
        #1、构建少量提示词模板
        examples = [
            {"input": "博睿智启上有关于AI应用开发的课程吗？", "output": "博睿智启上有哪些课程？"},
            {"input": "博小睿出生在哪个国家？", "output": "博小睿的个人经历是怎样的？"},
            {"input": "司机可以开快车吗？", "output": "司机可以做什么？"},
        ]
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt
        )
        #2、构建生成回退问题提示
        system_prompt = "你是一个世界问题的专家，你的任务是回退问题，将问题改述为更一般或者前置问题，这样更容易回答，请参考示例来实现。"
        prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            few_shot_prompt,
            ("human","{question}")
        ])
        # 3、构建生成回退问题的链
        chain  = (
            {"question":RunnablePassthrough()}
            |prompt
            |self.llm
            |StrOutputParser()
            |self.retriever
        )
        return chain.invoke(query)
# 3、构建向量数据库与检索器
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
retriever = db.as_retriever(search_type="mmr")
# 创建问题回退检索器
step_back_retriever = StepBackRetriever(
    retriever=retriever,
    llm=ChatOpenAI(model="gpt-4o-mini")
)
# 检索文档
docs = step_back_retriever.invoke("人工智能会改变世界吗？")

for doc in docs:
    print("===============================================")
    print(doc.page_content[:30])
client.close()
























