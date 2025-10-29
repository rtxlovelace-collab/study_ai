'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_app_callback_use.py
* @Time: 2025/9/1
* @All Rights Reserve By Brtc
'''
from typing import Any, Optional, Union
from uuid import UUID

import dotenv
from langchain_core.callbacks import BaseCallbackHandler, StdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
def rertriever_from_qa(query:str):
    print(f"我正在向量数据库里面检索用的问题:{query}")
    return f"我的名字叫博小睿， 喜欢篮球、rap、和唱跳"
class LLMOpsCallBackHandler(BaseCallbackHandler):
    """自定义LLMOps回调处理器"""
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        print("on_llm_start--->:", serialized)
        print("on_llm_start--->:", prompts)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        print("on_llm_new_token--->")

#1、构建提示词
prompt = ChatPromptTemplate.from_template(""" 
请根据用户的问题回答,可以参考对应上下文进行回答:
<context>
{context}
</context>
""")
# 3、构建大模型
llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
chain = RunnablePassthrough.assign(context=lambda x:rertriever_from_qa(x["query"]))|prompt|llm|parser
content = chain.stream({"query":"你好你是谁？"},
                       config={"callbacks":[StdOutCallbackHandler(), LLMOpsCallBackHandler()]}
                       )
for one in content:
    print(one)

