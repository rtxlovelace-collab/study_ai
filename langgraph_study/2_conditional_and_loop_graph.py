'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 2_conditional_and_loop_graph.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
import json
from typing import TypedDict, Annotated, Any
import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph
from pydantic import Field, BaseModel
from typing_extensions import Literal
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
class ChatBotState(TypedDict):
    messages:Annotated[list, add_messages]
bot_tools = [google_serper, dalle]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(bot_tools)
def chat_bot(state:ChatBotState):
    """大语言模型节点"""
    ai_messages = llm_with_tools.invoke(state["messages"])
    return {"messages":[ai_messages]}
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
# 编译
graph_run = graph_builder.compile()
state = graph_run.invoke({"messages":[("human","2024年北京半程马拉松的前三名成绩是多少？")]})
for message in state["messages"]:
    print(f"消息类型:{message.type}")
    if hasattr(message,"tool_calls") and len(message.tool_calls) > 0:
        print(f"工具调用参数:{message.tool_calls}")
    print(f"消息内容:{message.content}")
    print("=======================================================")

