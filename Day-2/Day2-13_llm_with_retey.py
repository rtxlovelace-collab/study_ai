#添加抖动机制
from itertools import chain

from langchain_core.runnables import RunnableLambda

counter = -1
def func(x):
    global counter
    counter += 1
    print("counter:", counter)
    return x/counter
#重试抖动机制
#调用失败时，会重试两次
chain = RunnableLambda(func).with_retry(stop_after_attempt=2)
#调用chain对象的invoke方法并传入3
resp = chain.invoke(3)
print(resp)