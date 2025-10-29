from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter

#构建递归文档分割器
loader = UnstructuredMarkdownLoader("./text.md")
documents = loader.load()

#构建分割器
text_splitter = CharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=50,
    add_start_index=True,
)
#分割文档
chunks = text_splitter.split_documents(documents)
print(chunks)
#打印分割结果
for one in chunks:
    print(f"块内容大小{len(one.page_content)},元数据{one.metadata}")