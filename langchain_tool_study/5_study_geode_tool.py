'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_study_geode_tool.py
* @Time: 2025/9/11
* @All Rights Reserve By Brtc
'''
import json
import os
from typing import Any, Type

import dotenv
import requests
from langchain_core.tools import BaseTool
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
                        return json.dumps(weather_data)
                return f"获取{city}的天气失败请仔细排查原因"
        except Exception as e:
            print(f"{e}")

gaode_weather = GaodeWeatherTool()
print(gaode_weather.invoke({"city":"长沙"}))