'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 4_study_base_tool_use.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
from typing import Type, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a:int = Field(description="第一个参数")
    b:int = Field(description="第二个参数")

class MultiplyTool(BaseTool):
    """乘法计算工具"""
    name:str = "multipy_tool"
    description :str ="将两个参数相乘"
    args_schema:Type[BaseModel] = CalculatorInput

    def _run(self, a:int, b:int) -> int:
        return a * b
cal_tool = MultiplyTool()

#打印相关的工具信息
print(f"名称:{cal_tool.name}")
print(f"描述:{cal_tool.description}")
print(f"参数:{cal_tool.args}")
print(f"直接返回:{cal_tool.return_direct}")
print(cal_tool.invoke({"a":2,"b":3}))