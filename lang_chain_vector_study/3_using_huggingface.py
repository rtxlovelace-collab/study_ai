'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 3_using_huggingface.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
query_vector = embeddings.embed_query("你好我叫 博小睿 你叫什么？")
print(query_vector)
print(len(query_vector))