import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
#prompt
prompt=ChatPromptTemplate.from_template("生成一个小于{x}的随机数")
#llm
'''
可调节的大模型温度，configurable_fields是定义哪些字段（也就是模型的参数）可以在后续进行配置。
ConfigurableField 是 LangChain 中用于定义可配置字段的类。
'''
llm = ChatOpenAI(model = "gpt-4o-mini").configurable_fields(
    temperature = ConfigurableField(id = "llm_temperature",name = "大语言模型温度参数",
                                   description = "用于调整大模型温度"))

#chain
chain = prompt | llm | StrOutputParser()
#调用chain
context = chain.invoke({"x": 1000},config={"configurable": {"llm_temperature": 1}})
print(context)