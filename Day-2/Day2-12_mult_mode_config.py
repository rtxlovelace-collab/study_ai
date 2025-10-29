import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

#prompt and llm
prompt = ChatPromptTemplate.from_template("请如实回答{qurey}")
llm = ChatOpenAI(model = "gpt-3.5-turbo").configurable_alternatives(
    ConfigurableField(id = "llm"),
    gpt4m= ChatOpenAI(model = "gpt-4o-mini"),
    gpt4= ChatOpenAI(model = "gpt-4")

)
#chain
chain = prompt | llm | StrOutputParser()
#动态回退模型配置
#run
content = chain.invoke({"qurey":"你是gpt几"},config={"configurable": {"llm":"gpt4"}})
print(content)