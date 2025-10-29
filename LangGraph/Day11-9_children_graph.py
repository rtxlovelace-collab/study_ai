from typing import Annotated, Any

import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

dotenv.load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini")
#谷歌搜索工具
class GoogleSerperArgsSchema(BaseModel):
    query:str =Field(description = "图片生成")#函数内置提示词，即使写错也不会报错或者不调用

google_serper =  GoogleSerperRun(
    api_wrapper=GoogleSerperAPIWrapper(),
    args_schema=GoogleSerperArgsSchema,
)

def reduce_str(left: str | None, right: str | None) -> str:
    if right is not None and right != "":
        return right
    return left

class AgentState(TypedDict):
    '''智能体状态'''

    query:Annotated[str,reduce_str]#原始问题
    live_content: Annotated[str, reduce_str]  # 直播文案
    xhs_content: Annotated[str, reduce_str]  # 小红书文案
    '''Annotated 是 Python 3.9 + 引入的标准库类型（来自typing模块），
    语法格式为 Annotated[基础类型, 元数据1, 元数据2, ...]，
    作用是给 “基础类型” 附加额外的描述性信息（元数据），
    这些元数据不直接影响代码运行，但可被工具、框架或自定义逻辑读取使用。'''

class LiveAgentState(AgentState,MessagesState):
    '''直播文案智能体状态'''
    pass

class XHSAgentState(AgentState):
    '''小红薯智能体状态'''
    pass

def chatbot_live(state:LiveAgentState) ->Any:
    '''直播文案智能体机器人'''
    #创建提示模板+链应用
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "你是一个拥有10年经验的直播文案专家，请根据用户提供的产品整理一篇直播带货脚本文案，如果在你的知识库内找不到关于该产品的信息，可以使用搜索工具。"
        ),
        ("human", "{query}"),
        ("placeholder", "{chat_history}")
    ])

    chain = prompt | llm.bind_tools([google_serper])

    #调用链并生成ai消息
    ai_message= chain.invoke({"query":state['query'],"chat_history":state['messages']})

    return {
        "messages":  ai_message,# 追加新消息
        "live_content":ai_message.content
    }

#创建直播文案智能体子图
live_agent = StateGraph(LiveAgentState)
#添加节点
live_agent.add_node("chatbot_live",chatbot_live)
live_agent.add_node("tools",ToolNode([google_serper]))
#添加边
live_agent.set_entry_point("chatbot_live")
live_agent.add_conditional_edges("chatbot_live",tools_condition)
live_agent.add_edge("tools","chatbot_live")

def chatbot_xhs(state:XHSAgentState,) ->Any:
    '''小红薯智能体机器人'''
    #创建提示模板+链应用
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个小红书文案大师，请根据用户传递的商品名，生成一篇关于该商品的小红书笔记文案，注意风格活泼，多使用emoji表情。"),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    #调用链并生成内容更新状态
    return {"xhs_content":chain.invoke({"query":state['query']})}

#创建小红薯智能体子图
xhs_agent = StateGraph(XHSAgentState)
#添加节点
xhs_agent.add_node("chatbot_xhs",chatbot_xhs)
#添加边
xhs_agent.set_entry_point("chatbot_xhs")
xhs_agent.set_finish_point("chatbot_xhs")

#创建整体入口图
def parallel_node(state:AgentState,config:RunnableConfig) -> Any:
    return state

#创建主图
main_graph = StateGraph(AgentState)
#添加节点
main_graph.add_node("chatbot_xhs",xhs_agent.compile())
main_graph.add_node("chatbot_live",live_agent.compile())#这里可以看到，我们使用了.compile()方法，而不是chatbot_live，因为我们需要将chatbot_live作为一个节点添加到主图中，而不是直接调用它。
main_graph.add_node("parallel_node",parallel_node)

#边
main_graph.set_entry_point("parallel_node")
main_graph.add_edge("parallel_node","chatbot_xhs")
main_graph.add_edge("parallel_node","chatbot_live")
main_graph.set_finish_point("chatbot_xhs")
main_graph.set_finish_point("chatbot_live")

agent = main_graph.compile()

print(agent.invoke({"query":"RTX 4060显卡"}))



