import os

import dotenv
import weaviate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey
from weaviate.proto.v1.v6300.v1.weaviate_pb2_grpc import Weaviate

dotenv.load_dotenv()
#构建文档分析器
loader = UnstructuredMarkdownLoader('./数据文档/01.项目API文档.md')
#构建分割器
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n","。|！|？","\.\s|\!\s|\?\s",";|；|：|，|、|"],
    is_separator_regex = True,
    chunk_size=500,
    chunk_overlap=50,
    # add_special_index=True,
)
'''（RecursiveCharacterTextSplitter），主要功能是将长文本分割成指定大小的片段，适用于处理大段文本（如文档内容）以便后续进行语义分析或嵌入处理。
主要参数解析：
separators：分割符列表，按优先级从高到低排列，优先使用靠前的分隔符分割文本。这里包含了中英文常见的段落分隔符（如\n\n表示空行）、换行符、中文标点符号、带空格的英文标点，以及中文标点等，兼顾中英文文本分割需求。
is_separator_regex=True：表明分隔符使用正则表达式模式，支持用|表示 “或” 逻辑（如。|！|？表示遇到这三个符号都可分割）。
chunk_size=500：每个文本片段的最大长度（通常以字符数或 token 数计）。
chunk_overlap=50：相邻片段间重叠的长度，用于保持上下文连贯性，避免分割破坏语义完整性。'''

documents = loader.load()
chunks = text_splitter.split_documents(documents)
#构建向量数据库
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_KEY"))
)
'''
weaviate.connect_to_weaviate_cloud() 是 Weaviate 向量数据库 Python 客户端库中的一个函数，用于连接到 Weaviate Cloud 服务（WCD，Weaviate Cloud Database）。
其主要作用是建立与云端 Weaviate 实例的连接，通常需要传入云端实例的 URL（如 https://your-instance.weaviate.network）和 API 密钥（API key）等认证信息。
调用该函数后会返回一个连接对象，通过这个对象可以对云端的 Weaviate 数据库进行操作，比如创建类、插入数据、执行向量搜索等。
它是 Weaviate 客户端库中用于云服务连接的便捷方法，简化了与云端实例的对接流程，无需手动配置复杂的连接参数。

'''
embedding = OpenAIEmbeddings(model = "text-embedding-3-small")
db = WeaviateVectorStore(
    client=client,
    embedding=embedding,
    index_name = "DataSetTest",
    text_key = "text",
)
db.add_documents(chunks)
#转换检索器
retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k": 10, "score_threshold": 0.5}
)
#检索
results = retriever.invoke("如何创建项目API文档？")
for one in results:
    print(one)
client.close()