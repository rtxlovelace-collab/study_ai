import uuid
from typing import Any,List,Optional,Iterable,Type

import dotenv
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


#自定义向量数据库
class MemoryVectorStore:
    store :dict= {}#在内存中储存向量

    def __init__(self,embedding:Embeddings, **kwargs):
        # 初始化

        self._embedding = embedding
        # 构造


    def add_texts(self,texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        #将数据添加到内存向量数据库中
        #判断metadatas和texts的长度是否一致
        if len(metadatas)!= len(texts) or metadatas is None or texts is None:
            raise ValueError("元数据格式必须与文本数据保持一致")

        #将文本转换为向量
        embeddings =self._embedding.embed_documents(texts)

        #生成uuid
        ids = [str(uuid.uuid4()) for text in texts]

        #将原始文本，向量，元数据，id构建成字典并储存
        for idx,text in enumerate(texts):
            self.store[ids[idx]] = {"text":text,
                                    "vector":embeddings[idx],
                                    "metadata":metadatas[idx]
                                    if metadatas is not None else {},
                                    "id": ids[idx]}
            return ids

    def similarity_search(query:str, k: int=4,**kwargs: Any) -> List[Document]:
        #执行相似性搜索
        #将query转换为向量
        embedding = self._embedding.embed_query(query)

        #循环遍历记忆储存，计算欧几里得距离
        result:list=[]
        for key ,record in self.store.items():
            distance = self._euclidean_distance(embedding,record["vector"])
            result.append({
                "distance":distance,
                **record,
            })

            #找到欧几里得距离最小的k条记录
            storted_result = sorted(result,key=lambda x: x["distance"])
            result_k = storted_result[: k]

            #循环构建文档列表并返回
            documents = [Document(page_content=item["text"],
                        text=record["text"],
                        metadata={**item["metadata"],
                        "score": item["distance"]})
                        for item in result_k]

            return documents

    @classmethod
    def from_texts(cls:Type["MemoryVectorStore"],texts:List[str],embedding:Embeddings,
                   metadata:Optional[List[dict]]=None,
                   **kwargs:Any) -> "MemoryVectorStore":
        #通过文本，嵌入模型元数据构建向量数据库
        memory_vector_store = cls(embedding=embedding,**kwargs)
        memory_vector_store.add_texts(texts,metadata)
        return memory_vector_store
    @classmethod
    def _euclidean_distance(cls,vec1,vec2)->float:
        #计算两个向量的欧几里得差距
        return np.linalg.norm(np.array(vec1)-np.array(vec2))

dotenv.load_dotenv()

#创建初始数据与嵌入模型
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
metadatas = [
    {"page": 1},
    {"page": 2},
    {"page": 3},
    {"page": 4},
    {"page": 5},
    {"page": 6, "account_id": 1},
    {"page": 7},
    {"page": 8},
    {"page": 9},
    {"page": 10},
]
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# 2.构建自定义向量数据库
db = MemoryVectorStore.from_texts(texts, embedding, metadatas)

# 3.执行检索
print(db.similarity_search("我养了一只猫，叫笨笨"))

#第一步，获取历史对话
#第二步，将对话转换为向量
#第三步将向量数据传入内存数据库
#第四步，当用户提问时