from typing import Any

import dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
dotenv.load_dotenv()
@tool
def complex_tool(int_arg:int,float_arg:float)->int:
    '''使用复杂工具进行复杂计算操作'''
    return int(int_arg*float_arg)
def try_except_tool(tool_args:dict,config:RunnableConfig)->Any:
    try:
        return complex_tool.invoke(tool_args,config)
    except Exception as e:
        return f"函数调用的参数如下：{tool_args}。\n错误信息：{e}。"
# 1.创建大语言模型并绑定工具
llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)
llm_with_tools = llm.bind_tools([complex_tool])
# 2.创建链并执行工具
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool
# 3.调用链
print(chain.invoke("使用复杂工具，对应参数为5和2.1"))
