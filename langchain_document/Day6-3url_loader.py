from langchain_community.document_loaders import WebBaseLoader
#网络文档加载器
loader = WebBaseLoader("https://www.4399.com")
'''WebBaseLoader 是 LangChain（一个用于构建大语言模型应用的开发框架）中的核心数据加载工具之一，
主要功能是从网页中提取文本数据，
为后续的文本处理（如分割、嵌入、检索等）提供原始素材，是连接 “网页信息” 与 “LLM 应用” 的关键组件。'''
documents = loader.load()
# print(documents)
print(documents[0].page_content,)#正文在索引0的page_content中
print(len(documents))