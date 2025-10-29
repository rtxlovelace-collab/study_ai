'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 14_study_Dalle_Output.py
* @Time: 2025/9/12
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

dalle = OpenAIDALLEImageGenerationTool(
    name="openai_dalle",
    api_wrapper=DallEAPIWrapper(model="dall-e-3")
)

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools=[dalle], tool_choice="openai_dalle")

chain = llm_with_tools|(lambda msg:msg.tool_calls[0]["args"]) | dalle

print(chain.invoke("请帮我生成一副美女的图片尽量二次元一点"))