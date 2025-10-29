'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 3_study_paralle_node.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
from time import sleep
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph

graph_builder = StateGraph(MessagesState)


def chat_bot(state:MessagesState)-> Any:
    print("这是开始了")
    return {"message":[AIMessage(content="你好我是OpenAI研发的机器人！")]}

def paralle_one(state:MessagesState)->Any:
    print("并行1",state)
    sleep(5)
    return{"messages":[HumanMessage(content="这是并行函数1")]}

def paralle_two(state:MessagesState)->Any:
    print("并行2",state)
    return {"messages":[HumanMessage(content="这是并行函数2")]}

def chat_end(state:MessagesState):
    print("结束", state)
    return {"messages": [HumanMessage(content="这是结束函数")]}
graph_builder.add_node("chat_bot",chat_bot)
graph_builder.add_node("paralle_one",paralle_one)
graph_builder.add_node("paralle_two",paralle_two)
graph_builder.add_node("chat_end",chat_end)

# 添加边
graph_builder.set_entry_point("chat_bot")
graph_builder.set_finish_point("chat_end")

graph_builder.add_edge("chat_bot", "paralle_one")
graph_builder.add_edge("chat_bot", "paralle_two")
graph_builder.add_edge("paralle_two", "chat_end")

run = graph_builder.compile()
print(run.invoke({"messages":[HumanMessage(content="你好你是")]}))