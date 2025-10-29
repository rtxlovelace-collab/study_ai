from langchain_community.document_loaders import UnstructuredMarkdownLoader
'''Markdown文档加载器使用技巧'''
loader = UnstructuredMarkdownLoader("./数据文档/01.项目API文档.md",mode="elements")
                                                                        #分片模式
document = loader.load()
#document的类型是元组，元组的第一个元素是文档的标题，第二个元素是文档的正文，第三个元素是文档的元数据
# print(document)
print(document[1].page_content)
print(len(document))