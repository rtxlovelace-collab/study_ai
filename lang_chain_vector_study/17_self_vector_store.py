'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 17_self_vector_store.py
* @Time: 2025/9/5
* @All Rights Reserve By Brtc
'''
import uuid
from typing import Iterable, Optional, Any
import dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VST
import numpy as np
from langchain_openai import OpenAIEmbeddings


class MemroryVectorStore(VectorStore):
    """自定义向量数据库"""
    store:dict ={}# 在内存中存储向量
    def __init__(self, embedding: Embeddings):
        """构造函数"""
        self._embeding = embedding

    def add_texts( self, texts: Iterable[str], metadatas: Optional[list[dict]] = None, *, ids: Optional[list[str]] = None, **kwargs: Any,) -> list[str]:
        """将数据添加到数据库中"""
        # 1、判断 metadata 跟 text 是不是 长度一致， 不一致就是错误
        if metadatas is None or len(metadatas) != len(texts):
            raise ValueError("元数据格式长度必须和文本数据保持一致")
        # 2、将文本转化成向量
        embeddings = self._embeding.embed_documents(texts)
        # 3、生成UUID
        ids = [str(uuid.uuid4()) for text in texts]
        #4、将原始文本 向量  元素据  id 构建成字典并存储
        for idx,text in enumerate(texts):
            self.store[ids[idx]] = {
                "id": ids[idx],
                "vector": embeddings[idx],
                "text":text,
                "metadata": metadatas[idx] if metadatas is not None else {}
            }
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """执行相似性检索"""
        # 1、将query转换成向量
        embedding = self._embeding.embed_query(query)

        #2、循环遍历记忆存储，计算欧几里得距离
        result:list=[]
        for key , record in self.store.items():
            distance = self._euclidean_distance(embedding, record["vector"])
            result.append({
                "distance":distance,
                **record
            })
        # 3、找到欧几里得最近的k条数据
        sorted_result = sorted(result, key=lambda x:x["distance"])
        result_k=sorted_result[:k]
        # 4、循环构建文档列表并返回
        documents = [
            Document(page_content=item["text"], metadata={**item["metadata"], "score":item["distance"]})
            for item in result_k
        ]
        return documents
    @classmethod
    def from_texts( cls: type["MemroryVectorStore"], texts: list[str], embedding: Embeddings, metadatas: Optional[list[dict]] = None, *, ids: Optional[list[str]] = None, **kwargs: Any, ) -> "MemroryVectorStore":
        """通过文本、嵌入模型、元素据构造向量数据库"""
        memory_vector_store = cls(embedding=embedding, **kwargs)
        memory_vector_store.add_texts(texts, metadatas)
        return memory_vector_store
    @classmethod
    def _euclidean_distance(cls, vec1, vec2)->float:
        """计算两个向量的欧式距离"""
        return float(np.linalg.norm(np.array(vec1) - np.array(vec2)))



dotenv.load_dotenv()

texts = [
    "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",
]
metadatas: list = [
    {"page": 1},
    {"page": 2},
    {"page": 3},
    {"page": 4},
    {"page": 5},
    {"page": 6},
    {"page": 7},
    {"page": 8},
    {"page": 9},
    {"page": 10},
]

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = MemroryVectorStore.from_texts(texts, embedding=embedding, metadatas=metadatas)
print(db.similarity_search("我养了一只猫"))