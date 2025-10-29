'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 6_using_qianfan_model.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.embeddings import QianfanEmbeddingsEndpoint

dotenv.load_dotenv()

embeddings = QianfanEmbeddingsEndpoint()
query_vector = embeddings.embed_query("你好我叫博小睿， 请问你叫什么？")

print(query_vector)
print(len(query_vector))