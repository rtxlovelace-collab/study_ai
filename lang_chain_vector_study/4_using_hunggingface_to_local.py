'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_using_hunggingface_to_local.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = "neuml/pubmedbert-base-embeddings",
                                    cache_folder= "./embeddings/")

query_vector = embeddings.embed_query("你好我叫 博小睿 你叫什么？")
print(query_vector)
print(len(query_vector))