from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter,Language

#代码分割器
#加载文档
loader = UnstructuredFileLoader("Day6-6-self_loader.py")
text_splitter = RecursiveCharacterTextSplitter.from_language(
    Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

docs = loader.load()
chunks= text_splitter.split_documents(docs)
for one in chunks:
    print(f"快内容大小{len(one.page_content)},元数据{one.metadata}")
    print("输出内容：",one.page_content)