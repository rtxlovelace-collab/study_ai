'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 1_study_duckduckgo_search.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.utils.function_calling import convert_to_openai_tool

search = DuckDuckGoSearchRun()

print(search.invoke("中国93大阅兵有那些新的装备？"))
print(f"工具名字:{search.name}")
print(f"工具描述:{search.description}")
print(f"工具参数:{search.args}")
print(f"是否直接返回:{search.return_direct}")
print(f"转换成OpenAI工具调用描述信息:{convert_to_openai_tool(search)}")