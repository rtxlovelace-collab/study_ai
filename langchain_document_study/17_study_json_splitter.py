'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 17_study_json_splitter.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
import requests
from langchain_text_splitters import RecursiveJsonSplitter
#1、获取并加载json
url = "https://api.smith.langchain.com/openapi.json"
json_data = requests.get(url).json()
#2、递归json 分割
text_splitter = RecursiveJsonSplitter(max_chunk_size=300)
#3、分割json数据并创建文档
json_chunks = text_splitter.split_json(json_data=json_data)
chunks = text_splitter.create_documents(json_chunks)
for chunk in chunks:
    print(chunk)
print(len(chunks))