'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 13_study_langurage_splitter.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# 1、加载文档
loader = UnstructuredFileLoader("./6_study_self_loader.py")
text_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON,
    chunk_size = 500,
    chunk_overlap = 50,
    add_start_index = True,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:
    print(f"块内容大小{len(chunk.page_content)}, 元素据:{chunk.metadata}")
    print(chunk.page_content)