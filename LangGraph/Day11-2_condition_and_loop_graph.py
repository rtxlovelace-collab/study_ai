import json
from typing import TypedDict, Annotated, Any, Literal

import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

dotenv.load_dotenv()
#创建谷歌搜索工具
class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")

google_serper = GoogleSerperRun(
    name="google_serper",
    description=(
        "一个低成本的谷歌搜索API。"
        "当你需要回答有关时事的问题时，可以调用该工具。"
        "该工具的输入是搜索查询语句。"
    ),
    args_schema=GoogleSerperArgsSchema,
    api_wrapper=GoogleSerperAPIWrapper(),
)

#定义工具列表与大模型
tools = [google_serper]
llm = ChatOpenAI(model = "gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)#给llm绑定工具

class ChatBotState(TypedDict):
    '''图状态数据结构，类型是字典'''
    messages: Annotated[list,add_messages]

def chat_bot(state:ChatBotState,):
    '''聊天机器人函数'''
    # 获取状态里储存的消息列表数据并传递给LLM
    ai_message = llm_with_tools.invoke(state['messages'])
    # 返回更新/生成的状态
    return {"messages": [ai_message], "node_name": 'node_name'}

def tool_executor(state:ChatBotState,):
    '''工具调用执行节点'''
    #构建工具名字映射字典
    tools_by_name= {tool.name: tool for tool in tools}

    #提取最后一条消息里的工具调用信息
    tool_calls = state['messages'][-1].tool_calls

    #遍历工具调用信息，执行相应的工具
    messages = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        #执行工具并将工具结果添加到消息列表中
        messages.append(ToolMessage(
            tool_call_id=tool_call["id"],
            content = json.dumps(tool.invoke(tool_call["args"])),
            name = tool_call["name"],
        ))

        #返回更新状态信息
        return {"messages": messages}


def route(state:ChatBotState,):
    '''动态选择工具执行或者结束'''
    #获取生成的最后一条消息
    last_message = state['messages'][-1]
    #检测消息是否存在tool_calls参数，如果是则执行`工具路由`
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_executor"
    #否则执行`聊天机器人`
    return END


# 1.创建状态图，并使用GraphState作为状态数据

graph_builder = StateGraph(ChatBotState)

# 2.添加节点
graph_builder.add_node("llm", chat_bot)
graph_builder.add_node("tool_executor", tool_executor)

# 3.添加边
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("tool_executor", "llm")
graph_builder.add_conditional_edges("llm", route)

# 4.编译图为Runnable可运行组件
graph = graph_builder.compile()

# 5.调用图架构应用
state = graph.invoke({"messages": [("human", "2024年北京半程马拉松的前3名成绩是多少")]})

for message in state["messages"]:
    print("消息类型: ", message.type)
    if hasattr(message, "tool_calls") and len(message.tool_calls) > 0:
        print("工具调用参数: ", message.tool_calls)
    print("消息内容: ", message.content)
    print("=====================================")