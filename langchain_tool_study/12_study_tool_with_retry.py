'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 12_study_tool_with_retry.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
from typing import Any, Dict

import dotenv
from langchain_core.messages import ToolCall, AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()
class CustomToolException(Exception):
    """自定义异常"""
    def __init__(self, tool_call:ToolCall, exception:Exception)->None:
        super().__init__()
        self.tool_call = tool_call
        self.exception = exception
@tool
def complex_tool(int_arg:int, float_arg:float, dict_arg:Dict)->int:
    """使用复杂工具进行复杂操作"""
    print("参数:", dict_arg)
    return int(int_arg*float_arg)
def tool_custom_exception(msg:AIMessage, config:RunnableConfig)->Any:
    print("============tool_custom_exception========================")
    print(msg.tool_calls[0]['args'])
    try:
        return complex_tool.invoke(msg.tool_calls[0]['args'], config)
    except Exception as e:
        raise CustomToolException(msg.tool_calls[0], e)
def exception_to_message(inputs:dict)->dict:
    print("=============exception_to_message=======================")
    # 提取错误信息
    exception = inputs.pop("exception")
    #2、将历史消息添加到原始输入中，以便模模型能够 修正错误
    messages = [
        AIMessage(content="", tool_calls=[exception.tool_call]),
        ToolMessage(tool_call_id=exception.tool_call["id"], content=str(exception.exception)),
        HumanMessage(content="最后一次调用工具引发了异常，请因该是参数不够，请增加一个字典参数{'a':6}")
    ]
    inputs['last_output']=messages
    return inputs
prompt = ChatPromptTemplate.from_messages([
    ("human", "{query}"),
    ("placeholder","{last_output}")
])
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0).bind_tools(tools=[complex_tool],tool_choice="complex_tool")

chain = prompt|llm|tool_custom_exception
self_crrect_chain = chain.with_fallbacks([exception_to_message|chain], exception_key="exception")

print(self_crrect_chain.invoke({"query":"计算。5， 2.1"}))