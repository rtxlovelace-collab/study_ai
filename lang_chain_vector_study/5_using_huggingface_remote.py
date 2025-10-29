'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_using_huggingface_remote.py
* @Time: 2025/9/4
* @All Rights Reserve By Brtc
'''
import os

import dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

dotenv.load_dotenv()
embeddings = HuggingFaceEndpointEmbeddings(model = "sentence-transformers/all-MiniLM-L12-v2")
query_vector = embeddings.embed_query("你好我叫 博小睿 你叫什么？")
print(query_vector)
print(len(query_vector))