'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_study_url_loader.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://www.qq.com")
documents = loader.load()

print(documents)
print(len(documents))
