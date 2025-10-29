'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 8_study_blob_filesys.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import FileSystemBlobLoader
loader = FileSystemBlobLoader(".", show_progress=True)
for blob in loader.yield_blobs():
    print(blob.source)