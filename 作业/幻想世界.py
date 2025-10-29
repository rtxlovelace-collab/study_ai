import dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor, initialize_agent, AgentType
from langchain.memory import ConversationSummaryBufferMemory
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
dalle = OpenAIDALLEImageGenerationTool(
    name="openai-dalle",
    description="一个低成本的图片生成工具，当你需要生成图片的时候可以调用该工具",
    api_wrapper=DallEAPIWrapper(model="dall-e-3"))

with open("./作业引用.txt", "r", encoding="utf-8") as f:
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"世界观{f.read()}。你是这个世界的一个万能向导，你的职责是帮助用户了解这个奇妙的世界，并且根据更具上下文引导用户拯救这个世界。"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

llm = ChatOpenAI(model="gpt-4")
tools = [dalle]
memory = ConversationSummaryBufferMemory(return_messages=True, input_key="input", llm=llm,
                                         max_token_limit=100)  # 根据对话摘要进行记忆
# 加载记忆
#context = memory.load_memory_variables
# memory_summary=RunnablePassthrough.assign(history=RunnableLambda(memory_variable)),
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent,
                               tools=tools,
                               memory=memory,
                               verbose=True)

def ai_response(user):
    res = agent_executor.invoke(user)
    return res["output"]

while True:
    print("欢迎来到幻想的空岛世界，输入quit/exit退出游戏。")
    user=input("you:")
    if user == "exit" or user == "quit":
        print("再见，期待再次相遇。")
        break
    #调用ai模块
    # context =ai_response(user)
    context=ai_response({"input": user})
    print("AI：",context)