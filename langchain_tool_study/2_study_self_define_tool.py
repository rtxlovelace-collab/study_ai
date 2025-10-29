'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 2_study_self_define_tool.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a:int = Field(description="第一个参数")
    b:int = Field(description="第二个参数")
"""
@tool
def multipy(a:int, b:int)->int:
    将传递的两个参数相乘
    print(f"开始计算a*b={a*b}")
    return a*b
"""
@tool("multy_tool", args_schema=CalculatorInput, return_direct=True, description="参数描述")
def multipy(a:int, b:int)->int:
    print(f"开始计算a*b={a*b}")
    return a*b
# 打印工具的信息
print(f"名称:{multipy.name}")
print(f"描述:{multipy.description}")
print(f"参数:{multipy.args}")
print(f"直接返回:{multipy.return_direct}")
print(multipy.invoke({"a":2,"b":3}))