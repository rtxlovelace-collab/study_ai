import dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

dotenv.load_dotenv()
#编排提示词prompt
prompt = ChatPromptTemplate.from_template('''向数据查询{qurey}中...''')
#创建大模型
llm = ChatOpenAI(model = "gpt-4o-mini")
#创建链
chain = {"query":RunnablePassthrough()} | prompt | llm |StrOutputParser()
#调用链
print(chain.invoke("hello who are you"))
