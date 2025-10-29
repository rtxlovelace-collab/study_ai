'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 9_blob_general_blob_loader.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders.generic import GenericLoader

loader = GenericLoader.from_filesystem(".", glob="*.txt", show_progress=True)
for idx , doc in enumerate(loader.lazy_load()):
    print(f"对当前加载:{idx+1}个文件， 文件信息:{doc.metadata}")
