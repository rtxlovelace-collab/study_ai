'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_mult_vector_retriever.py
* @Time: 2025/9/10
* @All Rights Reserve By Brtc
'''
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
# 1、创建加载器、文本分割器并处理文档
loader = UnstructuredFileLoader("./eshop_goods.txt")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = loader.load_and_split(text_splitter)
#2、定义摘要生成链
summary_chain = (
    {"doc":lambda x:x.page_content}
    |ChatPromptTemplate.from_template("总结以下文档的内容:\n\n{doc}")
    |ChatOpenAI(model="gpt-4o-mini", temperature=0)
    |StrOutputParser()
)
#3、批量生成摘要与唯一标识
summaries = summary_chain.batch(docs, {"max_concurrency":5})
doc_ids = [str(uuid.uuid4()) for _  in enumerate(docs)]
#4、构建摘要文档
summary_docs = [
    Document(page_content=summary, metadata={"doc_id":doc_ids[idx]})
    for idx, summary in enumerate(summaries)
]
#5、构建文档数据库与向量数据库
byte_store = LocalFileStore("./mult-vectyor")
db = FAISS.from_documents(summary_docs, embedding=OpenAIEmbeddings(model="text-embedding-3-small"))
#6、构建多向量检索器
retriever = MultiVectorRetriever(
    vectorstore=db,
    byte_store=byte_store,
    id_key="doc_id"
)
#7、将摘要文档和原文档存储到数据库中
retriever.docstore.mset(list(zip(doc_ids, docs)))
#8、执行检索
search_res = retriever.invoke("推荐一些潮州特产")
for doc in search_res:
    print("=====================================================")
    print(doc.page_content[:30])