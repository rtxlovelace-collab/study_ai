import dotenv
from langchain_core.messages import RemoveMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from typer.cli import state

dotenv.load_dotenv()

llm = ChatOpenAI(model = "gpt-4o-mini")
#编写节点
def chat_bot(state:MessagesState):
    '''聊天机器人节点'''
    return {"messages":[llm.invoke(state["messages"])]}

def del_human_message(state:MessagesState):
    '''删除人类消息节点'''
    human_message = state["messages"][0]
    return {"messages":[RemoveMessage(id = human_message.id)]}
def update_ai(state:MessagesState):
    '''修改ai消息节点'''
    ai_message = state["messages"][0]
    return {"messages":[AIMessage(id = ai_message.id,content = "我是被修改后的点杰"+ai_message.content )]}

# 1.创建图构建器
graph_builder = StateGraph(MessagesState)
# 2.添加节点
graph_builder.add_node("chat_bot", chat_bot)
graph_builder.add_node("del_human_message", del_human_message)
graph_builder.add_node("update_ai", update_ai)
# 3.添加边
graph_builder.set_entry_point("chat_bot")
graph_builder.add_edge("chat_bot","del_human_message")
graph_builder.add_edge("del_human_message","update_ai")
graph_builder.set_finish_point("update_ai")

# 4.编译图
graph = graph_builder.compile()
# 5.调用图应用程序
print(graph.invoke({"messages":[("human","你好，你是谁")]}))
