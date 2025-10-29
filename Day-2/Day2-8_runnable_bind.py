import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
prompt=ChatPromptTemplate.from_messages([
    ("system","请回答问题"),
    ("human","{query}")
                    ])

llm = ChatOpenAI(model="gpt-3.5-turbo")
#动态更改模型
chain = prompt| llm.bind(model="gpt-4") | StrOutputParser()
content = chain.invoke({"query":"你是gpt几？"})
print(content)