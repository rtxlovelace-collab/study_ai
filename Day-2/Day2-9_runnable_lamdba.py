import random

from langchain_core.runnables import RunnableLambda


# def get_weather(location:str,unit:str=None,city:str=None)->str:
def get_weather(data:dict={}):#直接在（）里定义字典
    #根据传入的位置+温度单元获取对应天气的信息
    # print("location:",location)
    # print("unit:",unit)
    # print("city:",city)
    location = data.get("location")
    unit = data.get("unit")
    return f"{location}天气为{random.randint(10,40)}{unit}"
get_weather_runnable = RunnableLambda(get_weather).bind()#bind可以传递多个参数
resp = get_weather_runnable.invoke({"location":"长沙","unit":"°C"})#invoke只能传递一个参数，所以需要bind,如果想穿多个参数可以在方法里定义一个字典
print(resp)

