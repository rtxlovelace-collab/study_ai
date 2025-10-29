import os
from operator import itemgetter

import dotenv
import weaviate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore

dotenv.load_dotenv()

def format_qa_pair(question:str, answer:str)->str:
    '''格式化穿的问题＋答案'''
    return f"Question:{question}\nAnswer:{answer}".strip()

#定义分解的子问题的prompt
decomposition_prompt = ChatPromptTemplate.from_template(
"你是一个乐于助人的AI助理，可以针对一个输入问题生成多个相关的子问题。\n"
    "目标是将输入问题分解成一组可以独立回答的子问题或子任务。\n"
    "生成与以下问题相关的多个搜索查询：{question}\n"
    "并使用换行符进行分割，输出（3个子问题/子查询）:"
    )
#构建问题分解链
decomposition_chain = (
    {"question":RunnablePassthrough()}
    | decomposition_prompt
    |ChatOpenAI(model = "gpt-4o-mini", temperature=0)
    |StrOutputParser()
    | (lambda x: x.split("\n"))
)

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

#执行获取子问题
question = "关于LLMOPS的应用配置文档有哪些？"
sub_questions = decomposition_chain.invoke(question)

#构建迭代问答链
prompt = ChatPromptTemplate.from_template(
    """ 
        这是你需要回答的问题：
        ---
        {question}
        ---

        这是所有可用的背景问题和答案对：
        ---
        {qa_pairs}
        ---

        这是与问题相关的额外背景信息：
        ---
        {context}
        ---

        使用上述背景信息和所有可用的背景问题和答案对来回答这个问题：

        {question}
        """
)

chain = (
    { "context":itemgetter("question")| retriever,
        "question":itemgetter("question"),
        "qa_pairs":itemgetter("qa_pairs")
    }
    | prompt
    |ChatOpenAI(model = "gpt-4o-mini", temperature=0)
    |StrOutputParser()
    )

#循环遍历所有子问题并进行检索 + 回答
qa_pairs = ''
for sub_question in sub_questions:
    answer = chain.invoke({"question":sub_question,"qa_pairs":qa_pairs})
    qa_pairs += format_qa_pair(sub_question, answer) + "\n"
    print("问题：", sub_question)
    print("回答", answer)
print("所有问题的回答：",qa_pairs)
print("问题分解完成！".center(50,"="))
client.close()