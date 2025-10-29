'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 3_study_office_loader.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import UnstructuredExcelLoader, UnstructuredPowerPointLoader, \
    UnstructuredWordDocumentLoader

#加载 excel
#loader = UnstructuredExcelLoader("./test.xlsx")
#加载ppt
#loader = UnstructuredPowerPointLoader("./博睿智启公开课第二次.pptx")
# 加载word
loader = UnstructuredWordDocumentLoader("./介绍.docx")
docs = loader.load()
print(docs)
print(len(docs))