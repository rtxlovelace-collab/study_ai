'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 1_study_usetext_loader.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import TextLoader

loader = TextLoader("./eshop_goods.txt", encoding="utf-8")

docments = loader.load()

print(docments)
print(len(docments))
print(docments[0].metadata)
