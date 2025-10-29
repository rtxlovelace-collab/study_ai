'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 6_study_memory_saver.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
from typing import Annotated, TypedDict

import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import Field, BaseModel

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
tools = [google_serper, dalle]

# 创建大预言模型
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 定义一个检查点
check_point=MemorySaver()
config = {"configurable":{"thread_id":1}}
config_2 = {"configurable":{"thread_id":2}}
agent = create_react_agent(model=model, tools=tools,checkpointer=check_point)
print(agent.invoke(
    {"messages":[("human","你好我叫博小睿，我喜欢游泳打球，请问你喜欢什么呀？")]},
    config=config
))
print(agent.invoke(
    {"messages":[("human","你知道我叫什么吗？")]},
    config=config_2
))