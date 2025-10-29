'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO
* @File: 2_study_cache_embedding.py
* @Time: 2025/9/3
* @All Rights Reserve By Brtc
'''
from tkinter.font import names
import dotenv
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
import numpy as np
from numpy.linalg import norm
def cosine_similarity(vec1:list, vec2:list)->float:
    # 1、计算点积
    dot_product = np.dot(vec1, vec2)
    #2、计算向量侧长度
    len_v1 = norm(vec1)
    len_v2 = norm(vec2)
    #3、计算余弦相似度
    return dot_product/(len_v2*len_v1)
dotenv.load_dotenv()
#1、创建文本嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_with_cache = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    LocalFileStore("./cache/"),
    namespace = embeddings.model,
    query_embedding_cache=True,
)
#2、嵌入文本
query_vector = embedding_with_cache.embed_query("我叫博小睿，喜欢篮球")
#3、嵌入文本列表
document_vector = embedding_with_cache.embed_documents([
    "我叫博小睿,喜欢打篮球。",
    "这个喜欢打篮球的男人叫博小睿。",
    "博小睿说过, 大模型应用的核心就是搞提示词。"
])
print("向量1 -------> 向量2的相似度", cosine_similarity(document_vector[0], document_vector[1]))
print("向量1 -------> 向量3的相似度", cosine_similarity(document_vector[0], document_vector[2]))
print("向量2 -------> 向量3的相似度", cosine_similarity(document_vector[1], document_vector[2]))