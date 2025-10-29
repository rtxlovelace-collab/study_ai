from typing import List

import jieba.analyse
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import TextSplitter


#自定义词分割
class CustomTextSplitter(TextSplitter):
    '''自定义文本分割器，传入的分隔符和默认关键词是10个'''
    def __init__(self, seperator,top_k=10,**kwargs ):
        super().__init__(**kwargs)
        self._sperator = seperator
        self._top_k = top_k
    def split_text(self,text:str)->List[str]:
        '''分割传入的文本为字符列表'''
    #1，根据传入的分隔符分割传入的文本
    #2，提取分割出来的文本的每一段的关键词，个数为top_k个
    #3，将关键使用逗号进行拼接组岑成字符串列表并返回
        split_texts=text.split(self._sperator)
        text_keywords = []
        for split_text in split_texts:
            text_keywords.append(
                #jieba coming
                jieba.analyse.extract_tags(split_text,self._top_k)

            )
        return [','.join(keywords) for keywords in text_keywords]

#创建加载器与分割器
loader= UnstructuredFileLoader('./data.txt')
#自定义分割器，以换行符为分隔符，提取关键词个数为10个
text_splitter=CustomTextSplitter('\n\n')

#加载文档并分割
documents=loader.load()
chunks=text_splitter.split_documents(documents)

#循环遍历文档信息
for one in chunks:
    print(f'文档块内容：{one.page_content}')