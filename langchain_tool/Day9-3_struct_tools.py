from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="第一个数")
    b: int = Field(description="第二个数")

def multipy(a: int, b: int) -> float:
    '传递两个参数相乘'
    print(f"开始计算：{a}/{b}={a / b}")
    return int(a / b)

async def multipy_async(a: int, b: int)-> float:
    '传递两个参数相乘'
    print(f"开始计算：{a}+{b}={a + b}")
    return a + b

calculator=StructuredTool.from_function(
    func=multipy,
    coroutine=multipy_async,
    args_schema=CalculatorInput,
    name="计算器",
    description="计算两个数的乘积",
    return_direct=True)

#打印工具信息
print(calculator)
print(f"name:{calculator.name}")
print(f"description:{calculator.description},描述")
print(f"args:{calculator.args},参数")
print("直接返回: ", calculator.return_direct)
#调用工具
print(calculator.invoke({"a": 2, "b": 3}))