
from typing import Annotated,TypedDict

from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import dotenv


dotenv.load_dotenv()
#定义状态数据结构
class GraphState(TypedDict):
    '''图状态数据结构，类型为字典'''
    messages:Annotated[list,add_messages]
    node_name:str


#定义节点函数
def chat_bot(state:GraphState):
    '''聊天机器人函数'''
    #获取状态里储存的消息列表数据并传递给LLM
    ai_message = llm.invoke(state['messages'])
    #返回更新/生成的状态
    return {"message":[ai_message], "node_name":'node_name'}
#实例化LLM
llm = ChatOpenAI(model="gpt-4o-mini")

#状态图，并使用GraphState作为状态数据
graph_builder = StateGraph(GraphState)

#添加节点
graph_builder.add_node('llm',chat_bot)#添加聊天节点

#添加边
graph_builder.add_edge(START,'llm')
graph_builder.add_edge('llm',END)

#编译 图为runnable可运行组件
graph = graph_builder.compile()

#测试
print(graph.invoke({'messages':[("human","你好你是谁？")], 'node_name':'graph'}))