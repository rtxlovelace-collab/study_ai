'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 2_study_markdown_loader.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader

"""Markdown 加载器使用技巧"""
#loader = UnstructuredMarkdownLoader("./01.项目API文档.md")
loader = UnstructuredMarkdownLoader("./01.项目API文档.md", mode = "elements")
docs = loader.load()

print(docs)
print(docs[0].metadata)
print(len(docs))
