from typing import TypedDict, Annotated
from langgraph.prebuilt import ToolNode
import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

dotenv.load_dotenv()
#创建工具
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

class State(TypedDict):
    '''图状态数据结构，类型是字典'''
    messages: Annotated[list,add_messages]

def chat_bot(state:State,):
    '''聊天机器人函数'''
    # 获取状态里储存的消息列表数据并传递给LLM
    ai_message = llm_with_tools.invoke(state['messages'])
    # 返回更新/生成的状态
    return {"messages": [ai_message]}


def route(state:State, ):
    '''获取选择工具执行或者结束'''
    #获取生成的最后一条信息
    last_message = state['messages'][-1]
    #检测消息是否存在tool_calls参数，若是则执行工具路由
    if hasattr(last_message,"tool_calls") and len (last_message.tool_calls) >0:
        return "tools"
    return END


#创建状态图，并使用GraphState作为状态数据
graph_builder = StateGraph(State)

#添加节点
graph_builder.add_node("llm",chat_bot)
graph_builder.add_node("tools",ToolNode(tools=tools))

#添加边
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("tools", "llm")
graph_builder.add_conditional_edges("llm", route)

# 4.编译图为Runnable可运行组件
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer, interrupt_after=["tools"])

# 5.调用图架构应用
config = {"configurable": {"thread_id": 1}}
state = graph.invoke(
    {"messages": [("human", "2024年北京半程马拉松的前3名成绩是多少")]},
    config,
)
print(state)

# 6.修改图状态消息
graph_state = graph.get_state(config)
tool_message = ToolMessage(
    id=graph_state[0]["messages"][-1].id,
    tool_call_id=graph_state[0]["messages"][-2].tool_calls[0]["id"],
    name=graph_state[0]["messages"][-2].tool_calls[0]["name"],
    content="2024年北京半程马拉松的第一名为博小睿 01:59:40，第二名为博二睿成绩为02:04:16，第三名为博三睿02:15:17",
)
graph.update_state(config, {"messages": [tool_message]})

# 7.继续执行图
print(graph.invoke(None, config)["messages"][-1].content)

