from langchain_core.tools import tool
from pydantic import BaseModel, Field

'''自定义工具'''
class CalculatorInput(BaseModel):
    a: int = Field(description="第一个数")
    b: int = Field(description="第二个数")
    '''一个名为 CalculatorInput 的类，该类继承自 pydantic 库中的 BaseModel 类。
这是使用 Pydantic 进行数据验证和解析的典型写法。通过继承 BaseModel，CalculatorInput
 会自动获得数据验证、类型转换、数据解析等功能。'''
    '''
@tool
def multipy(a: int, b: int) -> float:
    传递两个参数相乘'
    print(f"开始计算：{a}*{b}={a * b}")
    return a * b
'''

@tool("multy_tool",
      description="传递两个参数相乘", #description参数必须存在否则报错，作用是工具的描述
      args_schema=CalculatorInput,#args_schema参数选择性存在，作用是工具的输入参数，不存在则表示默认参数
      return_direct=True )
def multipy(a: int, b: int) -> float:
    '''传递两个参数相乘'''
    print(f"开始计算：{a}*{b}={a * b}")
    return a * b
#打印工具的信息
print(f"name:{multipy.name}")
print(f"description:{multipy.description},描述")
print(f"args:{multipy.args},参数")
print("直接返回: ", multipy.return_direct)
print(multipy.invoke({"a": 5, "b": 8}))