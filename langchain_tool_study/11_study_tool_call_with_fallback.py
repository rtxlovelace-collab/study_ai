'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 11_study_tool_call_with_fallback.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
from typing import Any

import dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
@tool
def complex_tool(int_arg:int, float_arg:float, list_arg:list)->int:
    """使用复杂工具进行复杂计算操作"""
    print("我被调用了", list_arg)
    print(f"参数{int_arg}*{float_arg}")
    return int(int_arg*float_arg)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools([complex_tool])
better_llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools([complex_tool], tool_choice="complex_tool")
#2、创建链并执行工具
better_chain = better_llm |(lambda msg:msg.tool_calls[0]["args"])|complex_tool
chain = (llm|(lambda  msg:msg.tool_calls[0]["args"])|complex_tool).with_fallbacks([better_chain])
# 3、调用链
print(chain.invoke("算一下,5 2.dhadhd [1,2,3,4,5,6]"))