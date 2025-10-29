from typing import Type

from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="第一个数")
    b: int = Field(description="第二个数")


class MutiplyTool(BaseTool):
    '乘法工具'
    name:str = 'multiply_tool'
    description:str = '两个参数相乘'
    args_schema :Type[BaseModel] = CalculatorInput
    def _run(self,a:int,b:int):
        return a*b
cal_tool = MutiplyTool()

#打印工具信息
print(cal_tool)
print(f"name:{cal_tool.name}")
print(f"description:{cal_tool.description},描述")
print(f"args:{cal_tool.args},参数")
print("直接返回: ", cal_tool.return_direct)
#调用工具
print(cal_tool.invoke({"a": 2, "b": 3}))