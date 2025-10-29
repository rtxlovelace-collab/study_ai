
'''使用blob实现文档加载'''
from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from langchain_core.documents.base import Blob


class CustomParser(BaseBlobParser):
    def lazy_parse(self, blob: Blob, one_line=None) -> Iterator[Document]:
        # 自定义解析逻辑
        line_number = 0
        with blob.as_bytes_io() as stream:
            for line in stream:
                line_number += 1
                # 解析一行数据
                yield Document(
                    page_content=line,
                    metadata={"source":blob.source,"line_number":line_number}
                )

blob =Blob.from_path("./text.md")
parser = CustomParser()
docs = list(parser.lazy_parse(blob))

print(docs)
print(len(docs))
for doc in docs:
    print(doc.page_content)
