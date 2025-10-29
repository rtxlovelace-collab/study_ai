'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 19_self_define_splitter.py
* @Time: 2025/9/8
* @All Rights Reserve By Brtc
'''
from typing import List
import jieba.analyse
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import TextSplitter
class  CustomTextSplitter(TextSplitter):
    """自定义文本分割器， 传入的分隔符和默认的关键词是 10个"""
    def __init__(self, seperator:str, top_k:int =10, **kwargs):
        super().__init__(**kwargs)
        self._seperator = seperator
        self._top_k = top_k
    def split_text(self, text:str)->List[str]:
        """分割传入的文本为字符列表"""
        #1、根据传入的分隔符分割传入的文本
        #2、提取分割出来的文本每一段的关键词，个数为top_k个
        #3、将关键词使用逗号进行拼接组成字符串列表返回
        split_texts = text.split(self._seperator)
        text_keywords = []
        for split_text in split_texts:
            text_keywords.append(
                jieba.analyse.extract_tags(split_text, self._top_k)
            )
        return [",".join(keywords) for keywords in text_keywords]
# 1、创建加载器与分割器
loader = UnstructuredFileLoader("./data.txt")
text_splitter = CustomTextSplitter(seperator="\n\n")
#2、加载并分割
documents = loader.load()
chunks = text_splitter.split_documents(documents)
for chunk in chunks:
    print(f"块内容大小{chunk.page_content}, 元素据:{chunk.metadata}")












