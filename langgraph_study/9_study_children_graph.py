'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 9_study_children_graph.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
from typing import  Annotated, Any
from typing_extensions import TypedDict
import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

class GoogleSerperArgsSchema(BaseModel):
    query:str = Field(description="执行谷歌搜索的查询语句")
google_serper = GoogleSerperRun(
    name = "google_serper",
    description = "一个低成本的谷歌搜索API 工具, 当你需要回答关键实事的 时候可以调用该工具",
    api_wrapper=GoogleSerperAPIWrapper(),
    args_schema=GoogleSerperArgsSchema
)

def reduce_str(left:str|None, right:str|None)->str:
    if right is not None and right!="":
        return right
    return left

class AgentState(TypedDict):
    query:Annotated[str, reduce_str]
    live_content:Annotated[str, reduce_str]
    xhs_content:Annotated[str, reduce_str]

class LiveAgentState(AgentState, MessagesState):
    """直播带货之智能体状态"""
    pass
class XHSAgentState(AgentState, MessagesState):
    """小红书智能体状态"""
    pass

def chat_bot_live(state:LiveAgentState)->Any:
    """直播文案带货机器人"""
    prompt= ChatPromptTemplate.from_messages([
        ("system", "你是一个拥有10年经验的直播文案专家，请根据用户提供的产品整理一篇直播带货脚本文案，如果在你的知识库内找不到关于该产品的信息，可以使用搜索工具。"),
        ("human","{query}"),
        ("placeholder","{chat_history}")
    ])
    chain = prompt|llm.bind_tools([google_serper])

    ai_messages = chain.invoke({"query":state["query"], "chat_history":state["messages"]})
    return {
        "messages":[ai_messages],
        "live_content":ai_messages.content
    }
#创建直播文案子图
live_agent_graph = StateGraph(LiveAgentState)

live_agent_graph.add_node("chat_bot_live", chat_bot_live)
live_agent_graph.add_node("tools", ToolNode([google_serper]))

live_agent_graph.set_entry_point("chat_bot_live")
live_agent_graph.add_conditional_edges("chat_bot_live", tools_condition)
live_agent_graph.add_edge("tools",  "chat_bot_live")

def chatbot_xhs(state:XHSAgentState)->Any:
    """小红书文案机器人"""
    """直播文案带货机器人"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个小红书文案大师，请根据用户传递的商品名，生成一篇关于该商品的小红书笔记文案，注意风格活泼，多使用emoji表情。"),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return {"xhs_content":chain.invoke({"query":state["query"]})}

# 创建小红书子图
xhs_agent_graph = StateGraph(XHSAgentState)

xhs_agent_graph.add_node("chatbot_xhs", chatbot_xhs)
xhs_agent_graph.set_entry_point("chatbot_xhs")
xhs_agent_graph.set_finish_point("chatbot_xhs")

# 创建入口并添加节点
def parallel_node(state:AgentState)->Any:
    return state

agent_graph = StateGraph(AgentState)
agent_graph.add_node("parallel_node", parallel_node)
agent_graph.add_node("live_agent", live_agent_graph.compile())
agent_graph.add_node("xhs_agent", xhs_agent_graph.compile())
agent_graph.set_entry_point("parallel_node")
agent_graph.add_edge("parallel_node", "live_agent")
agent_graph.add_edge("parallel_node", "xhs_agent")
agent_graph.set_finish_point("live_agent")
agent_graph.set_finish_point("xhs_agent")
agent = agent_graph.compile()

print(agent.invoke({"query":"潮汕牛肉丸"}))
























