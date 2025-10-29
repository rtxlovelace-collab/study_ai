'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 13_pic_to_weather.py
* @Time: 2025/9/12
* @All Rights Reserve By Brtc
'''
import json
import os
from typing import Type, Any

import dotenv
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

dotenv.load_dotenv()

class GadeWeatherSchema(BaseModel):
    city:str = Field(description="需要查询天气的城市，如：广州，长沙，杭州等等")

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
                        print(weather_data)
                        return json.dumps(weather_data)
                return f"获取{city}的天气失败请仔细排查原因"
        except Exception as e:
            print(f"{e}")

# 1、构建prompt
prompt=ChatPromptTemplate.from_messages([
    ("human",[
        {"type":"text","text":"请获取上传图片中对应城市的天气信息！"},
        {"type":"image_url", "image_url":{"url":"{image_url}"}}
    ])
])

weath_prompt = ChatPromptTemplate.from_template(""" 
请整理传递给你的城市天气信息,并以用户能够于阅读的方式输出！
### 天气信息如下:
<weather>
{weather}
</weather>
""")
# 构建llm 并执行
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools=llm.bind_tools([GaodeWeatherTool()], tool_choice="llmops_gaode_weather_tool")

# 调用链并执行
chain = (
    {
        "weather":(
            {"image_url":RunnablePassthrough()}
            |prompt
            |llm_with_tools
            |(lambda msg:msg.tool_calls[0]["args"])
            |GaodeWeatherTool()
        )
    }|weath_prompt|llm|StrOutputParser()
)
print(chain.invoke("https://img1.baidu.com/it/u=644490943,1781886584&fm=253&fmt=auto&app=138&f=JPEG"))
