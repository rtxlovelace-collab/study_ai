import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from scripts.regsetup import description

dotenv.load_dotenv()

class QAExtra(BaseModel):
    '''一个问答键值对工具，传递对应的假设行性问题 + 答 案'''
    question: str=Field(description="假设性问题")
    answer: str=Field(description="假设性问题的答案")
"结构化输出"
llm = ChatOpenAI(model = 'gpt-4o-mini')
struct_llm =llm.with_structured_output(QAExtra,method="json_mode")#后面加上method="json_mode"输出格式改成json
'''用于约束模型输出格式的功能、参数或代码装饰器，核心目的是让 AI 不再输出自由文本（如段落式回答），
而是生成符合预设结构（如 JSON、字典、特定类实例等）的数据，方便后续程序直接解析和使用。'''
prompt = ChatPromptTemplate.from_messages([
    ("system", '''请从用户传递的query中提取出假设性的问题+答案。
               响应结果为json格式包含 question 和 answer 两个字段'''),#提示词也要明确告诉大模型输出的格式
    ("human", "{query}")
])

chain = {"query": RunnablePassthrough()} | prompt | struct_llm

print(chain.invoke("我叫慕小课，我喜欢打篮球，游泳"))