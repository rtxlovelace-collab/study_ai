'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 16_study_baseon_toolcall_agent.py
* @Time: 2025/9/12
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
dotenv.load_dotenv()
class GoogleSerperArgsSchema(BaseModel):
    query:str = Field(description="执行谷歌搜索的查询语句")

class DalleArgsSchema(BaseModel):
    query: str = Field(description="输入应该是生成图片的提示(prompt)")

google_serper = GoogleSerperRun(
    name = "google_serper",
    description = "一个低成本的谷歌搜索API 工具, 当你需要回答关键实事的 时候可以调用该工具",
    api_wrapper=GoogleSerperAPIWrapper(),
    args_schema=GoogleSerperArgsSchema
)
dalle = OpenAIDALLEImageGenerationTool(
    name="openai_dalle",
    api_wrapper=DallEAPIWrapper(model="dall-e-3"),
    args_schema=DalleArgsSchema
)
tools = [google_serper,dalle]
# 2、定义工具调用agent 提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpul assistant."),
    ("placeholder","{chat_history}"),
    ("human","{input}"),
    ("placeholder", "{agent_scratchpad}")
])
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)
agnent_ext = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agnent_ext.invoke({"input":"2025年AI会爆发吗？"}))