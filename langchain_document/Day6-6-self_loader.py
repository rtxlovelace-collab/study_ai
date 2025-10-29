from typing import Iterator, AsyncIterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


#自定义文档加载器
class SelfLoader(BaseLoader):
    """
    自定义文档加载器
    """
    def __init__(self, path: str):
        self.file_path = path
    def lazy_load(self) -> Iterator[Document]:
        """
        按行读取文档内容，并返回Document对象
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            #file_path是文件，后面可以自定义后缀名
            line_number = 0
            for line in f:
                yield Document(page_content=line,
                               metadata={'line_number': line_number,
                                          'file_path': self.file_path})
                line_number += 1
    async def alazy_load(self)-> AsyncIterator[Document]:
        """
        异步按行读取文档内容，并返回Document对象
        """
        import aiofiles
        async with aiofiles.open(self.file_path, 'r', encoding='utf-8') as f:
            line_number = 0
            async for line in f:
                yield Document(page_content=line,
                               metadata={'line_number': line_number,
                                         'file_path': self.file_path})
                line_number += 1

loader = SelfLoader('text.md').load()
for doc in loader:
    print(doc)
    print(doc.metadata)


