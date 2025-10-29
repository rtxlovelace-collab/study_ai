
from typing import TypedDict, Annotated, Any, Literal

import dotenv
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import  StateGraph,START,END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

dotenv.load_dotenv()

class DallEArgsSchema(BaseModel):
    query: str=Field(description="输入应该是生成图像的文本提示(prompt)")


dalle = OpenAIDALLEImageGenerationTool(
    name="openai_dalle",
    api_wrapper=DallEAPIWrapper(model="dall-e-3"),
    args_schema=DallEArgsSchema,
    )


class State(TypedDict):
    '''图状态数据结构，类型为字典'''
    messages:Annotated[list,add_messages]

tools = [dalle]
llm = ChatOpenAI(model = "gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State, ) -> Any:
    """聊天机器人函数"""
    # 1.获取状态里存储的消息列表数据并传递给LLM
    ai_message = llm_with_tools.invoke(state["messages"])
    # 2.返回更新/生成的状态
    return {"messages": [ai_message]}


def route(state: State,) -> Literal["tools", "__end__"]:
    """动态选择工具执行亦或者结束"""
    # 1.获取生成的最后一条消息
    ai_message = state["messages"][-1]
    # 2.检测消息是否存在tool_calls参数，如果是则执行`工具路由`
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    # 3.否则生成的内容是文本信息，则跳转到结束路由
    return END

#创建状态图，并使用GraphState作为状态数据
graph_builder = StateGraph(State)

#添加节点
graph_builder.add_node('llm',chatbot)
graph_builder.add_node('tools',ToolNode(tools=tools))

#添加边
graph_builder.add_edge(START,"llm")
graph_builder.add_edge("tools","llm")
graph_builder.add_conditional_edges("llm",route)

#添加断点
checkpointer=MemorySaver()
graph_run = graph_builder.compile(checkpointer=checkpointer, interrupt_before=["tools"])

# 5.调用图架构应用
config = {"configurable": {"thread_id": 1}}
state = graph_run.invoke(
    {"messages": [("human", "帮我生成一只穿着风衣的兔子")]},
    config=config,
)
print(state)

# 6.获取人类的提示
if hasattr(state["messages"][-1], "tool_calls") and len(state["messages"][-1].tool_calls) > 0:
    tool_calls = state["messages"][-1].tool_calls
    print("准备调用工具: ", tool_calls)
    human_input = input("如需调用工具请回复yes，否则回复no: ")
    if human_input.lower() == "yes":
        print(graph_run.invoke(None, config))
    else:
        print("图程序执行结束")

        '''这段代码的主要作用是判断是否需要调用工具，并根据用户输入决定是否执行工具调用：
首先通过hasattr检查state["messages"]列表中最后一条消息是否包含tool_calls属性，同时判断该属性的长度是否大于 0，以此确定是否存在待执行的工具调用。
如果存在工具调用，会先打印 "准备调用工具:" 并显示具体的工具调用信息。
然后通过input函数等待用户输入，询问是否要调用工具（输入 yes 或 no）。
如果用户输入 "yes"（不区分大小写），则执行graph_run.invoke(None, config)来调用工具。
如果用户输入不是 "yes"，则打印 "图程序执行结束"，终止工具调用流程。
这段代码实现了一个人机交互的工具调用确认机制，让用户可以在程序执行工具前进行人工确认，增加了流程的可控性。'''