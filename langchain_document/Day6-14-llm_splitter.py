import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_experimental.text_splitter import SemanticChunker


dotenv.load_dotenv()
#构建文档加载器

loader = UnstructuredFileLoader("./data.txt")
text_splitter = SemanticChunker(
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small"),
    sentence_split_regex = r"(?<=[。？！])",
    number_of_chunks = 10,
    add_start_index = True,
)

#加载文本与分割
documents = loader.load()
chunks = text_splitter.split_documents(documents)
print(chunks[0].page_content)
# for one in chunks:
#     print("="*50)
#     print(f"块大小：{len(one.page_content)},元数据：{one.metadata}")
