from langchain_community.document_loaders import (UnstructuredExcelLoader,
                                                  UnstructuredPowerPointLoader,
                                                  UnstructuredWordDocumentLoader)

'''office文档加载器'''
#加载excel文件
loader = UnstructuredExcelLoader("./数据文档/test.xlsx")
#加载word文件
# loader = UnstructuredWordDocumentLoader("./数据文档/介绍.docx")
#加载ppt文件
# loader = UnstructuredPowerPointLoader("./数据文档/博睿智启公开课第二次.pptx")

docs = loader.load()
print(docs)
print(docs[0].page_content)
print(len(docs))