from typing import Any

import dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
dotenv.load_dotenv()
@tool
def complex_tool(int_arg:int,float_arg:float,list_arg:list)->int:
    '''使用复杂工具进行复杂计算操作'''
    print(f"int_arg:{int_arg},float_arg:{float_arg},list_arg:{list_arg}")
    return int(int_arg*float_arg)

# 1.创建大语言模型并绑定工具，当当前大模型处理不了时会回退到better_llm的大模型


llm = ChatOpenAI(model = "gpt-3.5-turbo-16k",temperature=0)
better_llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0).bind_tools([complex_tool], tool_choice="complex_tool")
# 2.创建链并执行工具
better_chain = better_llm | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool
chain = (llm | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool).with_fallbacks([better_chain])
# 3.调用链
print(chain.invoke("使用复杂工具，对应参数为5和3.abcdefg  [1,2,3]"))
