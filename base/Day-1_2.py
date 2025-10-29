from _operator import itemgetter

import dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel


dotenv.load_dotenv()
def rertriever_from_qa(query:str):
    print(f'正在向向量数据库中查询{query}问题...')
    return "你好，我是ikuu机器人，喜欢唱跳rap篮球，很高兴为您服务。"
#构建提示词
prompt = ChatPromptTemplate.from_template('''
根据用户的问题回答问题，可以参考对应的上下文进行回答：
<context>
{context}
</context>
        ''')

#构建大模型
llm = ChatOpenAI(model = "gpt-4o-mini")
parser = StrOutputParser()
#编排并运行链
chain = RunnableParallel({
    "context":lambda x:rertriever_from_qa(x["query"]),
    "query": itemgetter("query")
})| prompt | llm | parser
#RunnableParallel：创建并行执行的组件，同时处理两个任务
#"context"：通过rertriever_from_qa检索器，根据输入的 "query" 获取相关上下文文档
#"query"：使用itemgetter("query")提取原始查询内容
content = chain.invoke({"query":"你好，你是谁"})
print(content