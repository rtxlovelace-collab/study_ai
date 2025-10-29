from langchain_community.document_loaders import UnstructuredFileLoader
#通用文档加载器
loader = UnstructuredFileLoader("./数据文档/test.xlsx")
'''UnstructuredFileLoader 是 LangChain（一个流行的大语言模型应用开发框架）中的核心文件加载工具类，
主要作用是将各种 “非结构化格式” 的文件内容提取为文本数据，
为后续的大模型处理（如问答、摘要、知识库构建）提供基础文本输入。'''
documents = loader.load()
print(documents)
print(len(documents))