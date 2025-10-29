import dotenv
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

dalle = OpenAIDALLEImageGenerationTool(
    name="openai-dalle",
    api_wrapper=DallEAPIWrapper(model="dall-e-3")

)

llm = ChatOpenAI(model="gpt-4o-mini",temperature=1)
llm_tools=llm.bind_tools(tools=[dalle],tool_choice="auto")

chain = llm_tools | (lambda msg: msg.tool_calls[0]["args"]) | dalle

print(chain.invoke("帮我画一个二次元美女，要可爱的，穿白丝，穿着白色连衣裙，长头发，年纪在16到25岁"))

#根据文字生成图片