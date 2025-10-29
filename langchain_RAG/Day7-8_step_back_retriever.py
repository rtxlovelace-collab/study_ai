#回退检索器
import os

import dotenv
import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_weaviate import WeaviateVectorStore

dotenv.load_dotenv()
class StepBackRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        '''根据传递的query执行问题回退并检索'''
        #构建少量提示词模板
        examples = [
            {"input": "博睿智启上有关于AI应用开发的课程吗？", "output": "博睿智启上有哪些课程？"},
            {"input": "博小睿出生在哪个国家？", "output": "博小睿的个人经历是怎样的？"},
            {"input": "司机可以开快车吗？", "output": "司机可以做什么？"},
            {"input": "你有什么建议给我吗？", "output": "你有什么想法？"},
        ]
        example_prompt = ChatPromptTemplate.from_messages([
            ("huiman","{input}"),
            ("ai","{output}"),
        ])
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,

        )
        #构建生成回溯的问题提示
        system_prompt = "你是一个世界问题专家，你的任务是回退问题，将问题改述为更简单的问题。这样子更容易回答，可以参考示例"
        prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            few_shot_prompt,
            ("human", "{question}"),
        ])
        #构建生成回退的链
        chain = (
            {"question":RunnablePassthrough()}
            |prompt
            |self.llm
            |StrOutputParser()
            |self.retriever
        )
        return chain.invoke(query)
#构建向量数据库与检索器
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=os.getenv("WEAVIATE_KEY"),

)
embedding = OpenAIEmbeddings(model = 'text-embedding-3-small')
wave_db=WeaviateVectorStore(
                        client=client,
                        embedding=embedding,
                        index_name="LlmopsDataSet",
                        text_key="text",
)

retriever = wave_db.as_retriever(search_type="mmr")
#构建回答回退检索器
step_back_retriever = StepBackRetriever(
    retriever=retriever,
    llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)
)
'''StepBackRetriever（回溯检索器）是一种基于 **“先抽象、后具体” 逻辑 ** 设计的检索工具，
核心目标是解决传统检索中 “因查询过于具体 / 局限，导致遗漏关键相关信息” 的问题，
常见于需要深度推理的问答场景（如复杂知识问答、技术问题解决等）。'''

#检索文档
documents = step_back_retriever.retrieve("地球会在未来会发生爆炸吗？")
print(documents)
print(len(documents))
client.close()
