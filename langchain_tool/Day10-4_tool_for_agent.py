import json
import os
from typing import Type

import dotenv
import requests
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
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


class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")

class DallEArgsSchema(BaseModel):
    query: str = Field(description="输入应该是生成图片的文本提示词（prompt）")

#定义工具与工具列表
google_serper = GoogleSerperRun(
    name = "google_serper",
    description="你是一个低成本的谷歌搜索api，当你需要回答有关事实时可以调用工具，该工具输入的是搜索查询语句",
    args_schema=GoogleSerperArgsSchema,
    api_wrapper = GoogleSerperAPIWrapper()
)
dalle = OpenAIDALLEImageGenerationTool(
    name = "openai-dall",
    api_wrapper = DallEAPIWrapper(model="dall-e-3"),
    args_schema=DallEArgsSchema,

)

tools = [google_serper, dalle,GaodeWeatherTool()]

#定义工具调用agent提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个乐于助人且能力强大的ai机器人，需要调用工具时可以调用工具"),
    ("placeholder","{chat_history}"),
    ("human","{input}"),
    ("placeholder","{agent_scratchpad}")
])

#创建大语言模型
llm = ChatOpenAI(model="gpt-4o-mini")

#创建agent与agent执行者
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)


while 1:
    input_text = input("输入：")
    if input_text == "exit":
        break
    print(agent_executor.invoke({"input":input_text}))