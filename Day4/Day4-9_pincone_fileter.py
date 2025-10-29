import dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from torch.fx.experimental.unification.dispatch import namespace

dotenv.load_dotenv()
# 首先，我们需要创建一个OpenAI客户端，并连接到OpenAI文本嵌入模型。
embedding=OpenAIEmbeddings(model='text-embedding-3-small')
# 我们需要创建一个Pincoen客户端，并连接到Pinecone向量库。
db =PineconeVectorStore(index_name="text",embedding=embedding,namespace="study")
#Pincoen是一个开源的云向量库，它可以将文本数据转换为向量，并将向量存储在Pinecone云向量库中。
# 它用于连接Pinecone集群并进行向量检索、相似度搜索等操作。

#单条件查询
resp=db.similarity_search_with_relevance_scores(
    "我养了一只叫笨笨的猫",
    filter={"page":{"$lte":5}},
    k=5
)
#带多条件的查询
resp1=db.similarity_search_with_relevance_scores(
    "我养了一只叫笨笨的猫",
    filter={"$or":[{"page":5},{"account_id":1}]},
    k=5
)
for one in resp1:
    print(one)

for one in resp:
    print(one)