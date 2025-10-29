from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter


# 1.构建Markdown字符 归递分割器并获取文档列表
loader = UnstructuredMarkdownLoader("./text.md")
documents = loader.load()

#构建分割器
text_splitter = CharacterTextSplitter(
    separator ="\n\n",
    chunk_size = 500,
    chunk_overlap=50,
    add_start_index=True,
)
'''
这段代码创建了一个CharacterTextSplitter实例，用于将文本分割成适合处理的片段。
参数解析：
separator = "\n\n"：使用两个换行符作为文本分割的标志
chunk_size = 500：每个文本片段的目标长度为 500 个字符
chunk_overlap = 50：相邻片段之间重叠 50 个字符，避免分割可能导致的语义断裂
add_start_index = True：为每个片段添加其在原始文本中的起始位置索引
'''
#分割文档
chunks = text_splitter.split_documents(documents)

#输出分割结果
for one in chunks:
    print(f"快内容大小：{len(one.page_content)},元数据:{one.metadata}")