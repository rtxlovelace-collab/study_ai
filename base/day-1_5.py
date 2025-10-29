from typing import Optional, Any, Union
from uuid import UUID

import dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.callbacks import StdOutCallbackHandler, BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from twisted.names.client import query
from twisted.trial._dist.stream import chunk

dotenv.load_dotenv()
#作业：这def里面写1，读一个txt文件，返回txt文件内容，更具txt文件的内容回答问题
def rertriever_from_qa(query:str):
    print(f'正在向向量数据库中查询{query}问题...')
    return "你好，我是ikuu机器人，喜欢唱跳rap篮球，很高兴为您服务。"

class LLMOpsCallBackHandler(BaseCallbackHandler):
    def on_llm_start(self,
                     serialized:dict[str, Any],
                     prompts:list[str],
                     *,
                     run_id:UUID,
                     parent_run_id:Optional[UUID]=None,
                     tags:Optional[list[str]]=None,
                     metadata:Optional[dict[str,Any]]=None,
                     **kwargs:Any,
                     ) -> Any:
                        print("on_llm_start-->:", serialized)
                        print("on_llm_start-->:", prompts)

    def on_llm_new_token(self,
        token: str,
        *,
        chunk:Optional[Union[GenerationChunk,ChatGenerationChunk]]=None,
        run_id: UUID,
        parent_run_id: Optional[UUID]=None,
        **kwargs: Any,
        ) -> Any:
        print("on_llm_new_token-->:",chunk)



#构建提示词
prompt = ChatPromptTemplate.from_template('''
根据用户的问题回答问题，可以参考对应的上下文进行回答：
<context>
{context}
</context>
        ''')
#构建大模型
llm = ChatOpenAI(model = "gpt-4o-mini")
parser =StrOutputParser()
chain=RunnablePassthrough.assign(context=lambda x:rertriever_from_qa(x["query"]))|prompt|llm|parser
context=chain.invoke({"query":"what is your name?"},

                     config={"callbacks":[StdOutCallbackHandler(),LLMOpsCallBackHandler()]})
for i in context:
    print("这是一条token",i)