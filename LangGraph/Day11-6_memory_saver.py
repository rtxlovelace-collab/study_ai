import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import Field, BaseModel

dotenv.load_dotenv()

class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")

class DallEArgsSchema(BaseModel):
    query: str = Field(description="输入应该是生成图片的文本提示词（prompt）")


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

tools = [google_serper, dalle]

#创建大模型
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

#定义一个检查点
check_point = MemorySaver()
agent=create_react_agent(model=model, tools=tools, checkpointer=check_point)
'''MemorySaver通常用于保存和恢复智能体的中间状态，实现对话历史的持久化或任务中断后的继续执行
，避免因流程中断而丢失上下文信息。

create_react_agent 这个工具函数，生成一个能自主 “思考 - 行动” 的 AI 代理：
它会先分析任务（比如 “查今明两天北京天气并整理成表格”），
判断是否需要调用工具（如天气查询 API），
执行工具后再根据结果决定下一步（比如是否需要补充数据、是否直接整理答案），
这就是 “React 逻辑”（Reasoning + Acting）的核心。
'''

# 调用智能体并输出内容
print(agent.invoke(
    {"messages": [("human", "你好，我叫慕小课，我喜欢游泳打球，你喜欢什么呢?")]},
    config={"configurable": {"thread_id": 1}}
))

# 二次调用
print(agent.invoke(
    {"messages": [("human", "你知道我叫什么吗？")]},
    config={"configurable": {"thread_id": 1}}
))