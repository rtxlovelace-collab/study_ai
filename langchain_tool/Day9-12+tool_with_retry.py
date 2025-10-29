from typing import Any, Dict, Optional

import dotenv
from langchain_core.messages import ToolCall, AIMessage, HumanMessage
from langchain_core.messages.tool import  ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

class CustomToolException(Exception):
    '''自定义的工具错误异常'''
    def __init__(self, tool_call:ToolCall,exception:Exception) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.exception = exception

@tool
def complex_tool(int_arg:int,float_arg:float,dict_arg:dict)->int:
    '''一个复杂的工具函数'''
    print("开始进行计算",dict_arg)
    return int_arg * float_arg

def tool_custom_exception(msg:AIMessage,config:RunnableConfig)->Any:
    print("开始进行自定义异常处理")
    '''工具自定义异常处理'''
    try:
        return complex_tool.invoke(msg.tool_calls[0]["args"],config)
    except Exception as e:
        raise CustomToolException(msg.tool_calls[0],e)

def exception_to_messages(inputs:dict)->dict:
    '''异常处理函数，将异常信息转换为消息'''
    #从输入中提取错误信息
    exception = inputs.pop("exception")
    print("开始进行重试")
    #将历史消息添加到原始输入中，以便模型识别到它上一次工具调用中犯的一个错
    message = [
        AIMessage(content='',tool_calls=[exception.tool_call]),
        ToolMessage(tool_call_id=exception.tool_call["id"],content=str(exception.exception)),
        HumanMessage(content='最后一次工具调用引发了异常，发现是缺少一个字典，请增加一个字典参数 {"a":1}后重试')
    ]
    inputs['last_output']=message
    return inputs


#创建prompt
prompt=ChatPromptTemplate.from_messages([
    ("human","{query}"),
    ("placeholder","{last_input}")

])
#创建大预言模型并绑定工具
llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0).bind_tools(tools=[complex_tool])
#创建链并执行工具
chain = prompt | llm | tool_custom_exception
'''“tool_custom_exception” 从字面和技术场景常见含义来看，通常指工具（或第三方服务 / 功能模块）自定义的异常类型，'''
self_correcting_chain = chain.with_fallbacks([exception_to_messages | chain], exception_key="exception")

#调用自我纠正链完成任务
print(self_correcting_chain.invoke({"query":"使用复杂工具计算，参数5,2.5"}))
    #重试策略