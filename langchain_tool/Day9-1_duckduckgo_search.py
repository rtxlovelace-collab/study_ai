from langchain_community.tools import  DuckDuckGoSearchRun
from langchain_core.utils.function_calling import convert_to_openai_tool

search =DuckDuckGoSearchRun()
'''DuckDuckGo
它是基础载体 ——DuckDuckGo 本身是一款以 “隐私保护” 为核心卖点的搜索引擎，
它区别于谷歌、百度等会追踪用户搜索行为的平台，
不收集用户搜索历史、不做个性化推荐，这是该工具的底层特性基础。
“SearchRun” 可理解为 “搜索 + 执行” 的结合，推测其核心作用是：
自动调用 DuckDuckGo 完成搜索任务，并可能对搜索结果进行初步处理（如提取关键信息、筛选特定内容），
或联动其他工具执行后续操作'''

print(search.invoke("中国93大阅兵出现了哪些新装备？"))
print(f"工具名字：{search.name}")
print(f"工具描述：{search.description}")
print(f"工具参数：{search.args}")
print(f"是否直接返回结果？：{search.return_direct}")
print(f"转换成openai工具调用信息：{convert_to_openai_tool(search)}")
