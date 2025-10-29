'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 15_study_use_llm_splitter.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
# 1、构建文档加载器
loader = UnstructuredFileLoader("./data.txt")

text_splitter = SemanticChunker(embeddings=OpenAIEmbeddings(model = "text-embedding-3-small"),
                                sentence_split_regex=r"(?<=[。？！])",
                                number_of_chunks=10,
                                add_start_index=True)
# 2、加载文本与分割
documents = loader.load()
chunks = text_splitter.split_documents(documents)

for chunk in chunks:
    print("====================================================================")
    print(f"块内容大小{len(chunk.page_content)}, 元素据:{chunk.metadata}")
   # print(chunk.page_content)