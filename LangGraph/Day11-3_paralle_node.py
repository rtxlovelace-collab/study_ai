from time import sleep

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState

graph_bilder = StateGraph(MessagesState)


'''这是一个完整的线性节点，没有中途的跳转，所有节点都在一条线上'''
#编写节点
def chat_bot(state: MessagesState,):
    return {"message":[AIMessage(content = 'hello i am openai robot')]}

def parallel1(state: MessagesState,):
    print("并行1",state)
    return {"message":[HumanMessage(content = 'This is 1')]}
def parallel2(state: MessagesState,):
    sleep(3)
    print("并行2",state)

    return {"message":[HumanMessage(content = 'This is 2')]}

def chat_end(state: MessagesState,):
    print("聊天结束",state)
    return {"message":[HumanMessage(content = 'Chat is over')]}
#添加节点
graph_bilder.add_node("chat_bot", chat_bot)
graph_bilder.add_node("parallel1", parallel1)
graph_bilder.add_node("parallel2", parallel2)
graph_bilder.add_node("chat_end", chat_end)


#添加边
graph_bilder.set_entry_point('chat_bot')#起始边
graph_bilder.add_edge('chat_bot', 'parallel1')
graph_bilder.add_edge('parallel1', 'parallel2')
graph_bilder.add_edge('parallel2', 'chat_end')
graph_bilder.set_finish_point('chat_end')#结束边



#编译图
graph = graph_bilder.compile()
#调用图
print(graph.invoke({"messages":[HumanMessage(content = 'hello what your name')]}))