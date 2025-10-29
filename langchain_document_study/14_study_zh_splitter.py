'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 14_study_zh_splitter.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#1、构建文档分割器
loader = UnstructuredMarkdownLoader("./01.项目API文档.md")
documents = loader.load()

#2、构建分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "。|！|？",
        "\.\s|\!\s|\?\s",  # 英文标点符号后面通常需要加空格
        "；|;\s",
        "，|,\s",
        " ",
        ""
    ],
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True
)

chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    print(f"块内容大小{len(chunk.page_content)}, 元素据:{chunk.metadata}")