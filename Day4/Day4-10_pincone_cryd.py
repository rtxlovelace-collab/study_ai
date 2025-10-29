import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from torch.fx.experimental.unification.dispatch import namespace

dotenv.load_dotenv()

embedding=OpenAIEmbeddings(model='text-embedding-3-small')

db =PineconeVectorStore(index_name="text",embedding=embedding,namespace="study")

#删除
db.delete(ids=["3a2ba7fb-4302-48df-bed4-fa1b96d4d051"])
#更新数据
llmops_index = db.get_pinecone_index("llmops")
llmops_index.update(id="3a2ba7fb-4302-48df-bed4-fa1b96d4d051",
                    values=[],
                    namespace="study")
'''
首先通过db.get_pinecone_index("llmops")获取名为 "llmops" 的 Pinecone 索引实例，赋值给llmops_index变量。这里的db应该是一个已初始化的 Pinecone 数据库连接对象。
然后调用索引的update方法，对指定数据进行更新：
id="3a2ba7fb-4302-48df-bed4-fa1b96d4d051"：指定要更新的数据的唯一标识符
values=[]：将该数据的向量值更新为空列表（通常向量是模型生成的数值数组）
namespace="study"：指定操作所在的命名空间（Pinecone 中用于数据隔离的机制）
'''