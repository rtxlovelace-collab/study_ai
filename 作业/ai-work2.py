'''
作业:
请基于ReactAgent实现一个 聊天机器人
1、能够调用搜索工具  √
2、根据ip查询地址 √
3、查询天气 √
4、能够生成图片 √
5、能够正常聊天
6、支持记忆
7、能够检索向量数据的数据
循环对话(能够基于那个命令行聊天机器人实现)
'''
import dotenv
import json
import os
import openai
import time
from typing import Type, Optional, Dict, Any
import requests
import weaviate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_weaviate import WeaviateVectorStore
from prometheus_client.decorator import contextmanager
from pydantic import BaseModel, Field
dotenv.load_dotenv()
#查询天气
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




#生成图片
dalle = OpenAIDALLEImageGenerationTool(
    name="openai-dalle",
    description="一个低成本的图片生成工具，当你需要生成图片的时候可以调用该工具",
    api_wrapper=DallEAPIWrapper(model="dall-e-3"))
#搜索工具
google = GoogleSerperRun(
    name="google_serper_tool",
    description="一个低成本的谷歌搜索api工具，当你需要回答关键事实的时候可以调用该工具",
    api_wrapper = GoogleSerperAPIWrapper())


#ip查询地址
class IPQueryArgsSchema(BaseModel):
    IP: str = Field(description="需要查询IP的目标城市，例如：175.0.226.112")
class IPqueryTool(BaseTool):
    """根据传入的城市名查询天气"""
    name:str = "gaode_IP"
    description:str = "当你想根据IP查询地点的时候可以调用词工具。"
    args_schema: Type[BaseModel] = IPQueryArgsSchema
    def _run(self,*args,**kwargs):
        try:
            gaode_api_key = os.environ.get("GAODE_API_KEY")
            gaode_api_url = os.environ.get("GAODE_API_URL")
            if not gaode_api_key or not gaode_api_url:
                raise ValueError("请在.env文件中配置GAODE_API_KEY和GAODE_API_URL")
            else:
                #从参数中获取城市名
                IP = kwargs.get("IP")
                #请求高德服务获取adcode
                session = requests.session()
                #行政编码查询
                city_response = session.request(
                    method="GET",
                    url=f"{gaode_api_url}/ip?key={gaode_api_key}&ip={IP}",
                    headers={"Content-Type": "application/json;charset=UTF-8"},

                )
                city_response.raise_for_status()
                city_data = city_response.json()
                if city_data.get("info")=="OK":
                    return json.dumps(city_data)
                session.close()
                return f"获取{kwargs.get('city')}天气失败"
        except Exception as e:
            return f"获取{kwargs.get('city')}天气失败，原因：{e}"


# 连接到Weaviate服务
def connect_weaviate_cloud():
    """连接到Weaviate云服务并返回客户端"""
    client = None
    try:
        # 从环境变量获取Weaviate云服务信息
        # weaviate_url = os.getenv("WEAVIATE_URL")
        # weaviate_api_key = os.getenv("WEAVIATE_KEY")
        #
        # if not weaviate_url or not weaviate_api_key:
        #     print("请确保环境变量中设置了WEAVIATE_URL和WEAVIATE_KEY")
        #     exit(1)
        # 配置本地Weaviate客户端
        client = weaviate.connect_to_local(os.getenv("WEAVIATE_HOST"))
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        db = WeaviateVectorStore(client=client, index_name="DatasetTest",
                                 text_key="text", embedding=embedding, )
        return db,client
    except Exception as e:
        print(f"连接Weaviate云服务失败: {e}")
        print("请检查你的Weaviate云服务URL和API密钥是否正确")
        exit(1)


# 存储对话到Weaviate云服务
def store_conversation(db:WeaviateVectorStore,ai_out,user_input):
    """将对话内容存储到Weaviate服务"""
    # 存储到Weaviate，
    base = [f'Human:{user_input} AI:{ai_out}']
    db.add_texts(base)


# 检索相关的历史对话
def retrieve_relevant_conversations(db, query):
    search=db.similarity_search_with_relevance_scores(query)#根据输入内容的向量表示，在目标向量数据库 / 集合中找到最相似的结果，

    context=''
    for one in search:
        context+=one[0].page_content
    return context


# 生成回复
def generate_response():
    #工具列表
    tools=[google, IPqueryTool(), GaodeWeatherTool(),dalle]
    # 构建提示词
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个乐于助人且能力强大的ai机器人，需要调用工具时可以调用工具"),
        ("human", "{context}\n{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini")
    #调用工具（也有chain的作用）
    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor



# 聊天主函数
def chat_loop():
    """聊天机器人主循环"""
    print("欢迎使用带记忆功能的聊天机器人！输入'exit','quit','退出' 退出聊天。")
    db ,client = connect_weaviate_cloud()
    if not db:
        print("无法连接到向量数据库，退出程序")
        return
    # 只创建一次agent
    agent_executor = generate_response()
    while True:
        # 获取用户输入
        user_input = input("你: ")
        if user_input in ['exit','quit','退出']:
            print("再见！")
            client.close()
            exit(-1)
        # 检索相关历史对话
        print("正在检索相关历史对话...")
        context = retrieve_relevant_conversations(db, user_input)
        res = agent_executor.invoke({'context': context, 'input': user_input})#工具调用
        ai_out = res['output']
        print("AI:", ai_out)
        store_conversation(db,ai_out, user_input)#储存对话
        # 存储对话到Weaviate
        # ai_out=res['output']
        # print("AI:",ai_out)
        # base=[f'Human:{user_input} AI:{ai_out}']
        # store_conversation(db, base)
if __name__ == "__main__":
    chat_loop()


