'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 10_study_text_splitter_use.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import  CharacterTextSplitter

#1、构建文档分割器
loader = UnstructuredMarkdownLoader("./01.项目API文档.md")
documents = loader.load()

#2、构建分割器
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True
)

#3、分割文档列表
chuncks = text_splitter.split_documents(documents)

#4、输出信息
for chunck in chuncks:
    print(f"块内容大小{len(chunck.page_content)}, 元素据:{chunck.metadata}")

