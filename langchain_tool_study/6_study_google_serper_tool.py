'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 6_study_google_serper_tool.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
dotenv.load_dotenv()
google_serper = GoogleSerperRun(
    name = "google_serper",
    description = "一个低成本的谷歌搜索API 工具, 当你需要回答关键实事的 时候可以调用该工具",
    api_wrapper=GoogleSerperAPIWrapper()
)
print(google_serper.invoke("中国93阅兵有哪些新武器！"))