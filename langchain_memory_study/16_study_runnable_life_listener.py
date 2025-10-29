'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 16_study_runnable_life_listener.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import time
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langsmith.schemas import Run
def on_start(run_obj:Run, config:RunnableConfig)->None:
    print("========================================")
    print("on_star")
    print("run_obj", Run)
    print("config", RunnableConfig)

def on_error(run_obj:Run, config:RunnableConfig)->None:
    print("========================================")
    print("on_error")
    print("run_obj", Run)
    print("config", RunnableConfig)

def on_end(run_obj:Run, config:RunnableConfig)->None:
    print("========================================")
    print("on_end")
    print("run_obj", Run)
    print("config", RunnableConfig)

runnable = RunnableLambda(lambda x:time.sleep(x))
chain = runnable.with_listeners(on_start=on_start, on_error=on_error, on_end=on_end)
chain.invoke(2, config= {"configurable":{"name":"brtc_alltman"}})