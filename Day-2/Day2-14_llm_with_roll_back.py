import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
#chain and llm and prompt
prompt = ChatPromptTemplate.from_template("{query}")
#如果模型失效，自动回退，回退列表中可以放多个模型，会一个个尝试
llm = ChatOpenAI(model="gpt-9").with_fallbacks([ChatOpenAI(model="gpt-4"),ChatOpenAI(model="gpt-4o-mini")])
chain = prompt | llm |StrOutputParser()
#运行
content = chain.invoke({"query":"你是gpt几？"})
print(content)