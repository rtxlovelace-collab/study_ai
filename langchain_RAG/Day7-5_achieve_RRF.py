import os
from typing import List

import weaviate
from langchain.load import  loads, dumps
import dotenv
from langchain.retrievers import MultiQueryRetriever
from langchain_core.callbacks import CallbackManager, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from nltk.corpus.reader import documents
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()

class RAGFusionRetriever(MultiQueryRetriever):
    '''RAG多查询结果融合检索器'''
    k: int = 10

    def __init__(self, k=4, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def retrieve_documents(self,queries:List[str],
                           run_manager:CallbackManagerForRetrieverRun, )->List[List]:
        '''重写检索文档，返回二层嵌套的列表'''
        document = []
        for query in queries:
           docs = self.retriever.invoke(
                query,
               config={"callbacks": run_manager.get_child()},
           )
           document.append(docs)
        return document

    def unique_union(self, document: list[List]) -> list[Document]:
        '''使用set去重并合并文档列表，RRF算法对文档列表进行排序&融合'''
        #初始化一个字典，用于存储每个文档的得分
        fused_scores = {}
        #遍历每个查询对应的文档列表
        for docs in document:
            #内层遍历文档列表的到每个文档
            for rank,doc in enumerate(docs):
                #将文档使用langchain提供的dumps方法序列化为字符串
                doc_str = dumps(doc)
                #检测该字符串是否存在得分，不存在则赋值为0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                #计算多结果得分，排名越小越靠前，k为控制权重的参数
                fused_scores[doc_str] += 1/(rank+60)

        #提取得分并进行排序
        reranked_results = [
            (loads(doc), score)
            for doc,score in sorted(fused_scores.items(), key=lambda x:x[1], reverse=True)
            ]

        return [item[0] for item in reranked_results[:self.k]]

#构建向量数据库与检索器
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)

embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
db = WeaviateVectorStore(client= client,
                         index_name="DataSetTest",
                         text_key="text",
                         embedding=embedding)

retriever = db.as_retriever(search_type="mmr")
#创建多重查询检索器
rag_fusion_retriever = RAGFusionRetriever.from_llm(
    retriever=retriever,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
)

#执行检索
docs = rag_fusion_retriever.invoke("关于LLMOps应用配置的文档有哪些")
print(docs)
print(len(docs))
client.close()