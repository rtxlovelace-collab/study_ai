from langchain_community.document_loaders import TextLoader
#textloader类用于加载text文件
loader = TextLoader("数据文档/eshop_goods.txt", encoding="utf-8")
document = loader.load()
print("document:",document)
print("长度：",len(document))
print(document[0].metadata)