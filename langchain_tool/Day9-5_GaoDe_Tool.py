import json
import os
from typing import Type, Any

import dotenv
import requests
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.tools import BaseTool

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

gaode_tool = GaodeWeatherTool()
gaode_json=json.loads(gaode_tool.invoke({"city":"长沙"}))
print(gaode_json("forecasts"[0].city))

