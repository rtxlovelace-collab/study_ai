import json
import os
from typing import Type, TypedDict, Dict, Optional
import dotenv
from langchain import requests
from langchain_core.messages.tool import tool_call
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_core.tools import BaseTool, render_text_description_and_args
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import ToolMessage
'''
对于不支持调用函数的agent解决方法
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

class ToolCallRequest(TypedDict):
    '''工具调用请求字典'''
    name: str
    agruments:Dict[str, str]

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

def invoke_tool(
        tool_call_request: ToolCallRequest,
        config:Optional[RunnableConfig]=None,
):
    '''
    我们可以使用执行工具调用的函数，
    :param tool_call_request: 一个包含键值对的字典，名称必须与现有的工具名匹配，参数是该工具的参数
    :param config:这是LangChain使用的包含回调、元数据等信息的配置信息
    :return:请求工具的输出内容。
    '''
    tool_name_to_tool = {tool.name:tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["agruments"],config=config)




#提示词
system_prompt = '''你是基于OpenAI开发的聊天机器人，可以访问呢以下工具。
                {rendered_tools}
                根据用户的输入返回要使用到的工具名称，
                将结果作为拥有key为‘name’和值为‘arguments’的json块输出，name为要调用的工具的名称，
                `arguments`是一个字典，其中键对应于参数名称，值对应于参数请求的值。
    
'''

#利用提示词让大模型调用工具，结果以json格式输出
prompt = ChatPromptTemplate.from_messages([
 ("system", system_prompt),
    ("human", "{query}"),
]).partial(rendered_tools=render_text_description_and_args(tools))
#构建大模型
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
#构建链
chain = (
        {"query": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
        | RunnablePassthrough.assign(output=invoke_tool)
)
'''，RunnablePassthrough 是一个基础的 “透传型” 组件，核心作用是 不修改输入数据，
仅将其原样传递到下游流程，同时适配框架的 Runnable 接口规范（确保能与其他组件无缝串联）。'''

print(chain.invoke("马拉松世界记录是多少?"))
