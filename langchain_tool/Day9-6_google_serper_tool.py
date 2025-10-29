import dotenv
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper

from langchain_core.pydantic_v1 import BaseModel, Field

dotenv.load_dotenv()

google_serper = GoogleSerperRun(
    name="google_serper_tool",
    description="一个低成本的谷歌搜索api工具，当你需要回答关键事实的时候可以调用该工具",

    api_wrapper = GoogleSerperAPIWrapper()

)
print(google_serper.invoke("中国93阅兵有哪些新武器？"))