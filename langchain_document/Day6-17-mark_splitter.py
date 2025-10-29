from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#定义文本分割器和加载器
loader= UnstructuredFileLoader("./data.txt")
text_splitter = RecursiveCharacterTextSplitter(
separators=[
        "\n\n",
        "\n",
        "。|！|？",
        "\.\s|\!\s|\?\s",  # 英文标点符号后面通常需要加空格
        "；|;\s",
        "，|,\s",
        " ",
        ""],
    is_separator_regex = True,
    chunk_size=500,
    chunk_overlap=50,
    length_function = calculate_token_count
)


#加载文档并执行分割
doc = loader.load
chunks = text_splitter.split_documents(doc)

for one in chunks:
    print(f"快内容长度：{len(one.page_content)}，元数据： {one.metadata}")
