'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_study_del_human_message.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
from typing import Any

import dotenv
from langchain_core.messages import RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
def chat_bot(state:MessagesState)->Any:
    """聊天机器人节点"""
    return {"messages":[llm.invoke(state["messages"])]}
def delete_human_message(state:MessagesState)->Any:
    """删除人类消息节点"""
    human_message = state["messages"][0]
    #通过remove 消息来删除人类消息
    return {"messages":[RemoveMessage(id=human_message.id)]}
def update_ai_message(state:MessagesState)->Any:
    """修改AI消息节点"""
    ai_message = state["messages"][-1]
    return {"messages":[AIMessage(id = ai_message.id, content="我是被修改后的节点 : " + ai_message.content)]}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chat_bot", chat_bot)
graph_builder.add_node("update_ai_message", update_ai_message)
graph_builder.set_entry_point("chat_bot")
graph_builder.add_edge("chat_bot", "update_ai_message")
graph_builder.set_finish_point("update_ai_message")
graph = graph_builder.compile()
print(graph.invoke({"messages":[("human","你好你是？")]}))
