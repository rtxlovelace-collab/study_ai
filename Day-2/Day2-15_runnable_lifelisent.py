#生命周期监听
import time

from langchain_core.runnables import RunnableConfig, RunnableLambda, Runnable
from langsmith.schemas import Run


def on_start(run_obj:Run, config:RunnableConfig)->None:
    print("="*40)
    print("on_start")
    print("run_obj",Run)
    print("config",RunnableConfig)

def on_error(run_obj:Run, config:RunnableConfig)->None:
    print("="*40)
    print("on_error")
    print("run_obj",Run)
    print("config",RunnableConfig)

def on_end(run_obj:Run, config:RunnableConfig)->None:
    print("="*40)
    print("on_end")
    print("run_obj",Run)
    print("config",RunnableConfig)
#下面是被监听的函数
runnable = RunnableLambda(lambda x:time.sleep(x))
#监听的函数可以是lambda函数，也可以是自定义的函数
chain = runnable.with_listeners(on_start=on_start, on_error=on_error, on_end=on_end)
chain.invoke(2,config = {"configurable":{"name":"brtc_alltman"}})
