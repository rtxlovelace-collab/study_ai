import json
import os
from typing import Type
import dotenv
from langchain import requests
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import ToolMessage
'''
初级可以调用工具的agent
'''
dotenv.load_dotenv()
class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")

class GaodeWeatherArgsSchema(BaseModel):
    city: str = Field(description="需要查询天气预报的目标城市，例如：广州")


class GaodeWeatherTool(BaseTool):
    """根据传入的城市名查询天气"""
    name:str = "gaode_weather"
    description:str = "当你想询问天气或与天气相关的问题时的工具。"
    args_schema: Type[BaseModel] = GaodeWeatherArgsSchema

    def _run(self,*args,**kwargs):
        try:
            gaode_api_key = os.environ.get("GAODE_API_KEY")
            gaode_api_url = os.environ.get("GAODE_API_URL")
            if not gaode_api_key or not gaode_api_url:
                raise ValueError("请在.env文件中配置GAODE_API_KEY和GAODE_API_URL")
            else:
                #从参数中获取城市名
                city = kwargs.get("city")
                #请求高德服务获取adcode
                session = requests.session()
                #行政编码查询
                city_response = session.request(
                    method="GET",
                    url=f"{gaode_api_url}/config/district?key={gaode_api_key}&keywords={city}&subdistrict=0",
                    headers={"Content-Type": "application/json;charset=UTF-8"},

                )
                city_response.raise_for_status()
                city_data = city_response.json()
                if city_data.get("info")=="OK":
                    adcode = city_data.get("districts")[0].get("adcode")
                    #天气查询
                    weather_info= session.request(
                        method="GET",
                        url=f"{gaode_api_url}/weather/weatherInfo?key={gaode_api_key}&city={adcode}&extensions=all",
                        headers={"Content-Type": "application/json;charset=UTF-8"},)
                    weather_info.raise_for_status()
                    weather_data = weather_info.json()
                    if weather_data.get("info")=="OK":
                        return json.dumps(weather_data)


                session.close()
                return f"获取{kwargs.get('city')}天气失败"
        except Exception as e:
            return f"获取{kwargs.get('city')}天气失败，原因：{e}"

#定义工具列表
gaode_weather = GaodeWeatherTool()
google_serper = GoogleSerperRun(
    name="google_serper_tool",
    description="一个低成本的谷歌搜索api工具，当你需要回答关键事实的时候可以调用该工具",
    api_wrapper = GoogleSerperAPIWrapper()
)
tool_dict = {
    gaode_weather.name:gaode_weather,
    google_serper.name:google_serper,
    }
tools = [tool for tool in tool_dict.values()]

#创建permpt
prompt = ChatPromptTemplate([
    ("system","你是OpenAI开发的聊天机器人，可以帮助用户回答任何问题，必要时可调用相关工具。"),
    ("human",'{query}')

])

#创建大预言模型并绑定工具
llm=ChatOpenAI(model='gpt-4o-mini',temperature=0)
llm_with_tools=llm.bind_tools(tools=tools)

#创建链
chain = {"query":RunnablePassthrough()} | prompt | llm_with_tools

#解析输出
query = "长沙目前的天气怎么样，适合穿什么衣服出门？"
response = chain.invoke(query)
tool_calls = response.tool_calls

#判断是工具调用还是正常输出结果
if len(tool_calls)<=0:
    print("生成内容：",response.content)
else:
    #将历史的系统信息，人类消息，AI消息组合
    messages = prompt.invoke(query).to_messages()
    messages.append(response)
    #循环所有工具调用信息
    for tool_call in tool_calls:
        tool = tool_dict.get(tool_call.get('name'))
        print("正在执行的工具",tool.name)
        id = tool_call.get('id')
        content = tool.invoke(tool_call.get("args"))
        messages.append(ToolMessage(
            content=content,
            tool_call_id=id,
        ))
        print("输出内容: ", llm.invoke(messages))