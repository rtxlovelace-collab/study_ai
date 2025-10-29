'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 8_study_unsupport_tools_llm.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
import json
import os
from typing import Type, Any, Optional, TypedDict, Dict
import dotenv
import requests
from langchain_cohere.react_multi_hop.prompt import render_tool
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.tools import BaseTool, render_text_description_and_args
from langchain_openai import ChatOpenAI
from openpyxl.styles.builtins import output
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

class ToolCallRequest(TypedDict):
    """工具调用请求列表"""
    name:str
    arguments:Dict[str,Any]


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

def invoke_tool(tool_call_request:ToolCallRequest, config:Optional[RunnableConfig] = None):
    """可以使用的工具执行函数
    :tool_call_request, 一个包含键名和参数的字典
    :config,这个是lanchain 的回调元素据信息
    return 请求工具输出的内容
    """
    tool_name_to_tool = {tool.name : tool for tool in tools }
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)


system_prompt =""" 
你是一个由OpenAI开发的聊天机器人,可以访问以下一组工具。
以下是每个工具的名称和描述:
{rendered_tools}
根据用户输入,返回要使用的工具名称和输入。
将您的响应作为具有"name"和"arguments"JOSN块的返回。
"arguments"应该是一个字典，其中键为对应参数的名称，值对应请求的值
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}")
]).partial(rendered_tools = render_text_description_and_args(tools))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#构建链
chain = (
    {"query":RunnablePassthrough()}
    |prompt
    |llm
    |JsonOutputParser()
    |RunnablePassthrough.assign(output=invoke_tool)
)
print(chain.invoke("广州明天的天气如何？"))