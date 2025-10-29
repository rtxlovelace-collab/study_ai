#自定义检索器

from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class CustomRetriever(BaseRetriever):
    '''自定义检索器'''
    documents: List[Document]#
    k: int=5#指定返回的相关文档数量上限
    def _get_relevant_documents(self, query: str, *,
                                run_manager:CallbackManagerForRetrieverRun=None) -> List[Document]:
        '''传递query 返回跟query相关的文档'''
        matching_documents=[]#初始化空列表存储匹配的文档
        for doc in self.documents:#遍历文档列表中的每一个文档
            if query.lower() in doc.page_content.lower():#进行判断查询字符串是否包含在文档内容中
                #如果包含，则将该文档添加到匹配列表
                matching_documents.append(doc)
            if len(matching_documents) >= self.k:#进行判断如果匹配的文档到达预设值则返回结果
                return matching_documents
        return matching_documents

    '''
这段代码定义了一个名为CustomRetriever的自定义检索器类，它继承自BaseRetriever基类，
主要功能是从文档集合中检索和查询相关的文档。
代码解析如下：
类定义：CustomRetriever(BaseRetriever)表明这是一个基于基础检索器的自定义实现
类属性：
documents: List[Document]：存储要检索的文档集合，类型为Document对象的列表
k: int：指定返回的相关文档数量上限
核心方法：_get_relevant_documents是检索器的核心实现，接收查询字符串query，返回相关的文档列表
检索逻辑：
初始化空列表matching_documents存储匹配的文档
遍历文档集合中的每个文档
当匹配文档数量达到k时返回结果
检查查询字符串（忽略大小写）是否包含在文档内容（忽略大小写）中
如果包含，则将该文档添加到匹配列表
    '''

# 1.定义预设文档
documents = [
    Document(page_content="笨笨是一只很喜欢睡觉的猫咪", metadata={"page": 1}),
    Document(page_content="我喜欢在夜晚听音乐，这让我感到放松。", metadata={"page": 2}),
    Document(page_content="猫咪在窗台上打盹，看起来非常可爱。", metadata={"page": 3}),
    Document(page_content="学习新技能是每个人都应该追求的目标。", metadata={"page": 4}),
    Document(page_content="我最喜欢的食物是意大利面，尤其是番茄酱的那种。", metadata={"page": 5}),
    Document(page_content="昨晚我做了一个奇怪的梦，梦见自己在太空飞行。", metadata={"page": 6}),
    Document(page_content="我的手机突然关机了，让我有些焦虑。", metadata={"page": 7}),
    Document(page_content="阅读是我每天都会做的事情，我觉得很充实。", metadata={"page": 8}),
    Document(page_content="他们一起计划了一次周末的野餐，希望天气能好。", metadata={"page": 9}),
    Document(page_content="我的狗喜欢追逐球，看起来非常开心。", metadata={"page": 10}),
]

#创建检索器
retriever = CustomRetriever(documents=documents, k=3)

#调用检索器获取搜索结果并打印
retriever_documents = retriever.invoke("猫")
for one in retriever_documents:
    print(one.page_content)
print(retriever_documents)
print(len(retriever_documents))