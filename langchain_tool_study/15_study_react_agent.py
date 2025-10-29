'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 15_study_react_agent.py
* @Time: 2025/9/12
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import render_text_description_and_args
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
google_serper = GoogleSerperRun(
    name = "google_serper",
    description = "一个低成本的谷歌搜索API 工具, 当你需要回答关键实事的 时候可以调用该工具",
    api_wrapper=GoogleSerperAPIWrapper()
)
tools = [google_serper]
#2、拉取智能体 提示词模板
prompt = hub.pull("hwchase17/react")
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    tools_renderer=render_text_description_and_args
)
agent_exe =AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_exe.invoke({"input":"中国的房价还会持续下跌吗？"}))