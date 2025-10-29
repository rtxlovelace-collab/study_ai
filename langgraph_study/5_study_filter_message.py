'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 5_study_filter_message.py
* @Time: 2025/9/15
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain_core.messages import HumanMessage, AIMessage, trim_messages
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
dotenv.load_dotenv()
messages = [
    HumanMessage(content="你好，我叫博小睿，我喜欢游泳打篮球，你喜欢什么呢？"),
    AIMessage([
        {"type": "text", "text": "你好，博小睿！我对很多话题感兴趣，比如探索新知识和帮助解决问题。你最喜欢游泳还是篮球呢？"},
        {
            "type": "text",
            "text": "你好，博小睿！我喜欢探讨各种话题和帮助解答问题。你对游泳和篮球的兴趣很广泛，有没有特别喜欢的运动方式或运动员呢？"
        },
    ]),
    HumanMessage(content="如果我想学习关于天体物理方面的知识，你能给我一些建议么？"),
    AIMessage(
        content="当然可以！你可以从基础的天文学和物理学入手，然后逐步深入到更具体的天体物理领域。阅读相关的书籍，如《宇宙的结构》或《引力的秘密》，也可以关注一些优秀的天体物理学讲座和课程。你对哪个方面最感兴趣？"
    ),
]
llm = ChatOpenAI(model="gpt-4o-mini")
update_messages = trim_messages(
    messages,
    max_tokens=80,
    token_counter=llm,
    strategy="first",
    allow_partial=True,
    text_splitter = RecursiveCharacterTextSplitter(),
)
print(update_messages)