from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field,BaseModel
import dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from scripts.regsetup import description

dotenv.load_dotenv()






class RouteQuery(BaseModel):
    '''将用户问题查询射影到最相关的数据源'''
    data_source:Literal['python_docs',
                        'json_docs',
                        'golang_docs',
                        "c_docs"] = Field(description =
                                          "根据给定用户问题，选择哪个数据源最相关以回答他们的问题"
                                          )

def choose_route(result : RouteQuery):
        if 'python_docs' in result.data_source.lower():
            return "chain for python_docs"
        if 'json_docs' in result.data_source.lower():
            return "chain for json_docs"
        else:
            return "chain for golang_docs"


# 1.构建大语言模型并进行结构化输出
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)

# 2.创建路由逻辑链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个擅长将用户问题路由到适当的数据源的专家。\n请根据问题涉及的编程语言，将其路由到相关数据源"),
    ("human", "{question}")
])
router = {"question": RunnablePassthrough()} | prompt | structured_llm | choose_route

# 3.执行相应的提问，检查映射的路由
question = """为什么下面的代码不工作了，请帮我检查下：

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("中文")"""

# 4.选择不同的数据库
print(router.invoke(question))
