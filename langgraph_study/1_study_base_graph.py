'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 1_study_base_graph.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
from typing import TypedDict, Annotated

import dotenv
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import add_messages, StateGraph

dotenv.load_dotenv()

class ChatBotGraphState(TypedDict):
    """聊天消息就是状态,消息是一个字典， add_message 是一个归纳函数"""
    messages:Annotated[list, add_messages]
    node_name:str

llm = ChatOpenAI(model="gpt-4o-mini")

def chat_bot(state:ChatBotGraphState)->ChatBotGraphState:
    """聊天机器人实例化函数"""
    ai_messages = llm.invoke(state["messages"])
    return {"messages":[ai_messages], "node_name":"chat_bot"}
graph_builder = StateGraph(ChatBotGraphState)

# START ----->  ChatBot -------> END
graph_builder.add_node("llm",chat_bot) # 添加聊天节点

graph_builder.add_edge(START,"llm")# 链接 开始节点和大语言模型节点 边START|LLM|END
graph_builder.add_edge("llm",END)# 链接 大预言模型和结束节点边


graph_bin = graph_builder.compile()

print(graph_bin.invoke({"messages":[("human","你好你是？")], "node_name":"chat_bot"}))

