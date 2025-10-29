'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_unstructed_file_loader.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
"""通用文件加载器"""

from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader("./介绍.docx")
docs = loader.load()
print(docs)