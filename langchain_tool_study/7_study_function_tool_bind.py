'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 7_study_function_tool_bind.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
import json
import os
from typing import Type, Any
import dotenv
import requests
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()
class GadeWeatherSchema(BaseModel):
    city:str = Field(description="需要查询天气的城市，如：广州，长沙，杭州等等")

class GoogleSerperArgsSchema(BaseModel):
    query:str = Field(description="执行谷歌搜索的查询语句")

class GaodeWeatherTool(BaseTool):
    """根据传入城市名称,调用API获取城市的天气"""
    name:str = "llmops_gaode_weather_tool"
    description:str="当你想要查询天气的时候请调用这个工具"
    args_schema:Type[BaseModel] = GadeWeatherSchema

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        try:
            gaode_api_key = os.getenv("GAODE_API_KEY")
            gaode_api_url = os.getenv("GAODE_API_URL")
            if not gaode_api_key or not  gaode_api_url:
                return f"请配置正确的key{gaode_api_key} 和正确的url{gaode_api_url}"
            else:
                # 1、从参数中获取城市信息
                city = kwargs.get("city")
                #2、请求高德服务获取adcode
                session = requests.session()
                #3、行政编码查询
                city_response = session.request(
                    method="GET",
                    url = f"{gaode_api_url}/config/district?key={gaode_api_key}&keywords={city}&subdistrict=0",
                    headers = {"Content-Type": "application/json; charset=utf-8"}
                )
                city_response.raise_for_status()
                city_data = city_response.json()
                if city_data.get("info") == "OK":
                    ad_code = city_data["districts"][0]["adcode"]
                    #开始获取天气信息
                    weather_info = session.request(
                        method="GET",
                        url = f"{gaode_api_url}/weather/weatherInfo?key={gaode_api_key}&city={ad_code}&extension=all",
                        headers = {"Content-Type": "application/json; charset=utf-8"}
                    )
                    # 开始请求
                    weather_info.raise_for_status()
                    # 数据解析
                    weather_data = weather_info.json()
                    if weather_data.get("info") == "OK":
                        #返回最终的结果
                        return json.dumps(weather_data)
                return f"获取{city}的天气失败请仔细排查原因"
        except Exception as e:
            print(f"{e}")

# 1、自定义工具列表
gaode_weather = GaodeWeatherTool()
google_serper = GoogleSerperRun(
    name = "google_serper",
    description="一个低成本的谷歌搜索工具",
    args_schema = GoogleSerperArgsSchema,
    api_wrapper=GoogleSerperAPIWrapper()
)

tool_dict = {
    gaode_weather.name:gaode_weather,
    google_serper.name:google_serper,
}
tools = [tool for tool in tool_dict.values()]
#2、创建prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你室友OpenAI开发的机器人,可以帮助用户回答问题,必要时请调用工具帮助用户解答"),
    ("human","{query}")
])
#3、创建大预言模型并绑定的工具
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tool = llm.bind_tools(tools = tools)
#4、创建链应用
chain = {"query":RunnablePassthrough()}|prompt|llm_with_tool
#5、解析输出
query =("请问广州的天气如何？")
resp = chain.invoke(query)

tool_calls = resp.tool_calls
#6、判断是否是工具调用还是正常的结果输出
if len(tool_calls)<=0:
    print("生成内容:", resp.content)
else:#有工具调用
    # 7、将历史的系统消息、人类消息、ai消息进行组合
    messages = prompt.invoke(query).to_messages()
    messages.append(resp)
    #8、循环遍历所有工具调用信息
    for tool_call in tool_calls:
        tool = tool_dict.get(tool_call.get("name"))
        print("正在执行工具调用:", tool.name)
        id = tool_call.get("id")
        content = tool.invoke(tool_call.get("args"))
        print("工具输出:", content)
        messages.append(ToolMessage(
            content=content,
            tool_call_id= id
        ))
    print("输出内容:", llm.invoke(messages))

