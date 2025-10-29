'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 3_study_struct_tools.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
from PIL.PalmImagePlugin import Palm8BitColormapImage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a:int = Field(description="第一个参数")
    b:int = Field(description="第二个参数")

def multiply(a:int , b:int)->int:
    """同步将传递的两个参数相乘"""
    return a*b

async def amultiply(a:int , b:int)->int:
    """异步将传递的两个参数相乘"""
    return a*b
calculator = StructuredTool.from_function(
    func = multiply,
    coroutine=amultiply,
    name = "mult_tools",
    description="用于将两个整数相乘，包含异步和同步方法",
    return_direct=True,
    args_schema=CalculatorInput,
)

print(f"名称:{calculator.name}")
print(f"描述:{calculator.description}")
print(f"参数:{calculator.args}")
print(f"直接返回:{calculator.return_direct}")

print(calculator.invoke({"a":2,"b":3}))