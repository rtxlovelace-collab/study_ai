from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#中文文档归递分割器

#构建文档分割器
loader = UnstructuredMarkdownLoader("./数据文档/01.项目API文档.md")
documents = loader.load()

#构建分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "。|！|？",
        "\.\s|\!\s|\?\s",  # 英文标点符号后面通常需要加空格
        "；|;\s",
        "，|,\s",
        " ",
        ""
    ],is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True
)

#加载文档与分割
documents = loader.load()
chunks = text_splitter.split_documents(documents)
print(chunks)
print(chunks[0].page_content)
#输出信息
for one in chunks:
    print(f"块大小：{len(one.page_content)},元数据：{one.metadata}")
