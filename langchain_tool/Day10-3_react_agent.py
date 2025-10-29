import dotenv

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import render_text_description_and_args
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
google_serper = GoogleSerperRun(
    name="Google_Serper",
    description="你是一个低成本的谷歌搜索api工具，当你需要回答关键事实时可以调用该工具快速获取信息。",
    api_wrapper=GoogleSerperAPIWrapper(),
)
tools = [google_serper]
#拉取智能体 提示词模板
prompt = hub.pull("hwchase17/react")
llm = ChatOpenAI(model="gpt-4o-mini",temperature=1)
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    tools_renderer=render_text_description_and_args,

)

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

print(agent_executor.invoke({"input":"导致中国的生育率会越来越低的主要原因是什么？"}))

#利用agent生成提示词来生成回答，直到结果接近完美为止