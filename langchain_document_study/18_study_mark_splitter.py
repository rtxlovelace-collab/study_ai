'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 18_study_mark_splitter.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
import tiktoken
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def calculate_token_count(query:str)->int:
    """计算传入文本的tokens数量"""
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    return len(encoding.encode(query))

#1、定义文本分割器和加载器
loader = UnstructuredFileLoader("./data.txt")
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
    length_function=calculate_token_count
)
#加载文档并执行分割
documents = loader.load()
chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    print(f"块内容大小{len(chunk.page_content)}, 元素据:{chunk.metadata}")
