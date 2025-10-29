'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 6_hyde_retriever.py
* @Time: 2025/9/9
* @All Rights Reserve By Brtc
'''
import os
from typing import List

import dotenv
import weaviate
from colorama.ansi import clear_line
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()

class HyDeRetriever(BaseRetriever):
    """HYDE 回合策略检索器"""
    retriever: BaseRetriever
    llm:BaseLanguageModel
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """传递检索query实现hyde混合检索策略"""
        # 1、构建生成假设性文档的prompt
        prompt = ChatPromptTemplate.from_template(
            "请写一篇科学论文来回答这个问题\n"
            "问题:{question}\n"
            "文章："
        )

        #2、构建链应用
        chain = (
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
# 创建 Hyde 检索器
hyde_retriever = HyDeRetriever(
    retriever=retriever,
    llm=ChatOpenAI(model="gpt-4o-mini")
)

# 检索 文档
docs = hyde_retriever.invoke("关于LLMOPS的应用配置文档有哪些？")
for doc in docs:
    print("========================================================")
    print(doc.page_content[:30])
client.close()