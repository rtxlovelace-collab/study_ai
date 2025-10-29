'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 2_study_achieve_RRF.py
* @Time: 2025/9/9
* @All Rights Reserve By Brtc
'''
import os
from typing import List

import dotenv
import weaviate
from langchain.retrievers import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()
class RAGFusionRetriever(MultiQueryRetriever):
    """RAG 多查询结果融合检索器"""
    k:int = 4

    def __init__(self, k:int=4, **kwargs):
        super().__init__(**kwargs)
        self.k = k


    def retrieve_documents( self, queries: list[str], run_manager: CallbackManagerForRetrieverRun) -> List[List]:
        """重写检索文档，返回二层嵌套列表"""
        documents = []
        for query in queries:
            docs = self.retriever.invoke(
                query,config={"callbacks":run_manager.get_child()}
            )
            documents.append(docs)
        return documents

    def unique_union(self, documents: List[List]) -> List[Document]:
        """使用 RRF算法对文档列表进行拍寻&合并"""
        # 1、初始化一个字典， 用于存储每一个唯一的得分
        fused_score = {}
        # 2、遍历每个查询对应的文档列表
        for docs in documents:
            # 3、内层遍历文档列表得到每一个文档的得分
            for rank, doc in enumerate(docs):
                # 4、将文档使用langchain提供的dump工具抓换成字符串
                doc_str = dumps(doc)
                #5、检测该字符串是否再字典里面如果不在则赋值为0
                if doc_str not in fused_score:
                    fused_score[doc_str] = 0
                # 6、计算多结果得分， 排名越小越靠前， k为控制权经验参数，经验值为60
                fused_score[doc_str] += 1/(60 + rank)
        #7，所有文档进行才重新排序
        ranked_result = [
            (loads(doc), score)
            for doc, score in sorted(fused_score.items(), key=lambda x:x[1], reverse=True)
        ]
        return [item[0] for item in ranked_result[:self.k]]


# 构建向量数据库
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WAEVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = WeaviateVectorStore(client=client,
                         index_name="DataSetTest",
                         text_key="text",
                         embedding=embedding)

retriever = db.as_retriever(search_type = "mmr")

rag_rrf_retriever = RAGFusionRetriever.from_llm(
    retriever = retriever,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

# 执行检索
docs = rag_rrf_retriever.invoke("关于LLMOPS应用配置文档有哪些？")

for one in docs:
    print("================================================")
    print(one.page_content[:30])
