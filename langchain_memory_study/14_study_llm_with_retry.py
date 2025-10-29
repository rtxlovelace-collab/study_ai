'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 14_study_llm_with_retry.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
from langchain_core.runnables import RunnableLambda

counter  = -1
def func(x):
    global counter
    counter += 1
    print("counter:", counter)
    return x/counter

chain = RunnableLambda(func).with_retry(stop_after_attempt=2)

resp = chain.invoke(3)
print(resp)