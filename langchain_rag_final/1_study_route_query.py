'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 1_study_route_query.py
* @Time: 2025/9/10
* @All Rights Reserve By Brtc
'''
from typing import Literal

import dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()

class RouteQuery(BaseModel):
    """将用户问题查询映射到最相关的数据源"""
    data_source:Literal["python_docs", "json_docs", "golang_docs","c_docs"] = Field(
        description="根据给定用户的问题，选择那个数据源最相关以回答他们的问题"
    )

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)
#3、执行响应的提问，检查对应的路由
question = """  为什么下面的代码不工作了，请帮我检查一下：

int main(){
    prn("helll");
}
"""
res = structured_llm.invoke(question)
print(res.data_source)

