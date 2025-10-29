'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 8_study_update_graph_state.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
import json
from typing import TypedDict, Annotated, Literal, Any

import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph
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
tools = [google_serper, dalle]

class ChatBotState(TypedDict):
    messages:Annotated[list, add_messages]
bot_tools = [google_serper, dalle]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(bot_tools)

def chat_bot(state: ChatBotState) -> Any:
    """聊天机器人处理函数"""
    ai_message = llm_with_tools.invoke(state["messages"])
    return {"messages":[ai_message]}

def tool_exe(state:ChatBotState)->Any:
    tool_by_name = {tool.name:tool for tool in bot_tools}
    tool_calls = state["messages"][-1].tool_calls# 最后一条信息才是最新的信息
    messages = []
    for tool_call in tool_calls:
        tool = tool_by_name[tool_call["name"]]
        messages.append(ToolMessage(
            tool_call_id = tool_call["id"],
            content = json.dumps(tool.invoke(tool_call["args"])),
            name = tool_call["name"],
        ))
    return {"messages":messages}
def route(state:ChatBotState)->Literal["tool_exe", "__end__"]:
    """动态选择工具或者结束"""
    ai_messages = state["messages"][-1]
    if hasattr(ai_messages,"tool_calls") and len(ai_messages.tool_calls)>0:
        return "tool_exe"
    return END
# 1、创建图
graph_builder = StateGraph(ChatBotState)
# 添加节点
graph_builder.add_node("llm",chat_bot)
graph_builder.add_node("tool_exe", tool_exe)
# 添加边
graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", route)
graph_builder.add_edge("tool_exe", "llm")
# 添加断点
check_point = MemorySaver()
graph_run = graph_builder.compile(checkpointer=check_point, interrupt_after=["tool_exe"])
config = {"configurable":{"thread_id":1}}
state = graph_run.invoke({"messages":[("human","2024年北京半程马拉松的前三名是谁？")]}, config=config)
print(state)
graph_state = graph_run.get_state(config)
tool_messages = ToolMessage(
    id = graph_state[0]["messages"][-1].id,
    tool_call_id=graph_state[0]["messages"][-2].tool_calls[0]["id"],
    name = graph_state[0]["messages"][-2].tool_calls[0]["name"],
    content="2024年北京半程马拉松的第一名 博小睿， 第二名我博二睿， 第三名为博三瑞"
)
graph_run.update_state(config, {"messages":[tool_messages]})
print(graph_run.invoke(None, config=config)["messages"][-1].content)