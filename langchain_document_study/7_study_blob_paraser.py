'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 7_study_blob_paraser.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from typing import Iterator
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob

"""使用 blob 实现文档的夹加载"""
class CustomParaser(BaseBlobParser):
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        line_number = 0
        with blob.as_bytes_io() as f:
            for line in f:
                yield Document(page_content=line, metadata={"line_number": line_number})
                line_number += 1
blob = Blob.from_path("./test.txt")
parser = CustomParaser()
docs = list(parser.lazy_parse(blob))
for one in docs:
    print(one)