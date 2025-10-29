'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 6_study_self_loader.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from typing import Iterator, AsyncIterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

""" 
 实现自定义的文件加载器
 1、读取文件
 2、每一行都封装成一个Document
 3、继承Baseloader 
"""

class CustomDocumentLoader(BaseLoader):
    """自定义文档加载器"""
    def __init__(self, org_file_path: str) -> None:
        self._file_path = org_file_path
    def lazy_load(self) -> Iterator[Document]:
        """逐行去读取文件中的数据并使用 yield 返回"""
        with open(self._file_path, "r", encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(page_content=line,
                               metadata={"line_number": line_number,
                                         "source": self._file_path,}
                               )
                line_number += 1
    async def alazy_load(self) -> AsyncIterator[Document]:
        """异步逐行加载"""
        import aiofiles
        async with aiofiles.open(self._file_path, "r", encoding="utf-8") as f:
            line_number = 0
            async for line in f:
                yield Document(page_content=line,
                               metadata={"line_number": line_number,
                                         "source": self._file_path,}
                              )
                line_number +=1
loader = CustomDocumentLoader("./test.txt")
docs = loader.load()
for one in docs:
    print(one)
print(len(docs))