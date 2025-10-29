import dotenv
import numpy as np
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from numpy.linalg import norm
from langchain.embeddings import CacheBackedEmbeddings

def cosine_similarity(vec1:list, vec2:list) -> float:
    #计算点积
    dot_product=np.dot(vec1, vec2)
    #计算向量侧长度
    len_1=norm(vec1)
    len_2=norm(vec2)
    #计算余弦相似度
    return dot_product/(len_1*len_2)

dotenv.load_dotenv()

#创建文本嵌入的模型
embeddings =OpenAIEmbeddings(model = "text-embedding-3-small")
#创建向量库
embedding_with_cache= CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    LocalFileStore("./cache/"),
    namespace=embeddings.model,
    query_embedding_cache=True
)

#嵌入式文本，将文本转换为向量
# query_vector= embedding_with_cache.embed_query("i am jerry,like play the Minecraft")
# print(query_vector)
# print(len(query_vector))
#嵌入文本列表
document_vector=embedding_with_cache.embed_documents([
    "i am jerry,like play the Minecraft",
    "i am jerry,like play the Minecraft",
    "i am jack,like play guitar",
    "i am jane,like play computer",])



#计算相似度
print("向量1-》向量2的相似度：",cosine_similarity(document_vector[0],document_vector[1]))
print("向量1-》向量3的相似度：",cosine_similarity(document_vector[0],document_vector[2]))
print("向量2-》向量3的相似度：",cosine_similarity(document_vector[1],document_vector[2]))
print("向量1-》向量4的相似度：",cosine_similarity(document_vector[0],document_vector[3]))


