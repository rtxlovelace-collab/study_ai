import json
import os
from typing import Type, Any

import dotenv
import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field, BaseModel
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

#根据上传的图片输出图片中的城市天气信息
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

#构建prompt
prompt = ChatPromptTemplate.from_messages([
    ("human",[
        {"type":"text","text":"请获取上传图片中所示城市今天的天气，不要获取其他日期的天气。"},
        {"type":"image_url","image_url":{"url":"{image_url}"}},
              ])

    ])

weather_prompt = ChatPromptTemplate.from_template(
    '''请整理传递的城市天气信息，并用人类友好方式输出
    <weather>
    {weather}
    </weather>
    '''
)

#构建llm并绑定工具
llm = ChatOpenAI(model = "gpt-4o-mini")
llm_tool = llm.bind_tools(tools=[GaodeWeatherTool()],tool_choice="gaode_weather")

#创建链并运行
chain= (
    {"weather":(
        {"image_url":RunnablePassthrough()}
        |prompt
        |llm_tool
        |(lambda msg:msg.tool_calls[0]["args"])
        |GaodeWeatherTool()
    )}|weather_prompt | llm |StrOutputParser()

)

#
print(chain.invoke("https://img95.699pic.com/photo/50133/4935.jpg_wh860.jpg"))