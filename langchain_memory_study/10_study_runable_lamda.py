'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 10_study_runable_lamda.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import random

from langchain_core.runnables import RunnableLambda


def get_weather(data:dict)->str:
    """根据传入的位置 + 温度单位获取对应的天气信息"""
    print("data:", data)
    return f"{data}"

get_weather_runnable = RunnableLambda(get_weather)

resp = get_weather_runnable.invoke({"data":{"a":"a","b":"b"}})
print(resp)
