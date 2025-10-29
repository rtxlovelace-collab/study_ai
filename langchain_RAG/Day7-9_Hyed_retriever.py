import os

import dotenv
import weaviate
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore

dotenv.load_dotenv()

class HyedRetriever(BaseRetriever):
    '''HyDE混合策略检索器'''
    retriever:BaseRetriever
    llm:BaseLanguageModel

    def _get_relevant_docunments(self,query:str,
                                 *,run_manager:CallbackManagerForRetrieverRun)->list[Document]:
        '''传递检索query实现HyDE混合检索策略'''
        #构建生成假设性文档的peompt
        prompt = ChatPromptTemplate.from_template(
            "请写一篇科学论文来回答这个问题。\n"
            "问题：{question}\n"
            "文章："
        )

        #构建链应用
        chain= (
            {"question":RunnablePassthrough()}
            |prompt
            |self.llm
            |StrOutputParser()
            |self.retriever
        )

        return chain.invoke(query)

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=os.getenv("WEAVIATE_KEY"),

)
embedding = OpenAIEmbeddings(model = 'text-embedding-3-small')
retriever =client.as_retriever(search_type="mmr")#构建一个检索器
db=WeaviateVectorStore(
                        client=client,
                        embedding=embedding,
                        index_name="LlmopsDataSet",
                        text_key="text",
)
#构建Hyde检索器
hyde_retriever = HyedRetriever(
    retriever=retriever,
    llm=ChatOpenAI(model = 'gpt-4o-mini',temperature=0),
)
'''Hybrid Retriever”（混合检索器）核心是一种融合多种检索策略的信息检索组件，
广泛应用于大语言模型（LLM）的检索增强生成（RAG）架构、智能问答系统等场景'''
#检索文档
documents = hyde_retriever.invoke("关于LLMOps应用配置的相关文档有哪些？")
for one in documents:
    print("="*40)
    print(one.page_content[:30])
client.close()
