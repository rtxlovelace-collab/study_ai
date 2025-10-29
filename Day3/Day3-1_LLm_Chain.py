import dotenv
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
prompt = ChatPromptTemplate.from_template("讲一个{subject}的故事")
llm = ChatOpenAI(model="gpt-4o-mini")
chain = LLMChain(llm=llm, prompt=prompt)
'''LLMChain 是 LangChain 框架中的一个核心组件，
主要用于将大语言模型（LLM）与提示词模板（PromptTemplate）结合，构建一个完整的问答或文本生成流程。'''
print(chain.invoke("computer"))
lecel