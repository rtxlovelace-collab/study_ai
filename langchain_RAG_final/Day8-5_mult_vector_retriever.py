import uuid

import dotenv
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


dotenv.load_dotenv()
'''多向量索引'''
#创建加载器
loader = UnstructuredFileLoader("./数据文档/eshop_goods.txt")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs = loader.load_and_split(text_splitter)

#定义摘要生成链
summary_chain = (
    {"doc": lambda x:x.page_content}
    |ChatPromptTemplate.from_template("总结以下文档内容：\n\n{doc}。")
    |ChatOpenAI(model="gpt-4o-mini",temperature=0)
    |StrOutputParser()
)

#批量生成摘要与唯一标识
summaries = summary_chain.batch(docs,{"max_concurrency": 5})
doc_ids = [str(uuid.uuid4()) for _ in enumerate(docs)]

#构建摘要文档
summary_docs = [
    Document(page_content=summary,metadatas={"doc_id": doc_ids[idx]})
    for idx,summary in enumerate(summaries)
    ]

#构建文档数据库与向量库
dyte_store = LocalFileStore("./multy-vector")
db = FAISS.from_documents(
    summary_docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

#构建多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=db,
    byte_store=dyte_store,
    id_key="doc_id",
)

#将摘要文档和原文档储存到向量库中
retriever.docstore.mset(list(zip(doc_ids,docs)))

# 8.执行检索
search_docs = retriever.invoke("潮州有哪些特产?")
print(search_docs)
print(len(search_docs))