'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 3_study_basechat_memory.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from langchain.memory.chat_memory import BaseChatMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 把我们的聊天机器人 使用Basechatmemory 实现 记忆功能

memory = BaseChatMemory(
    input_key="query",
    output_key="output",
    return_messages=True
)

memory_var = memory.load_memory_variables({})
# 获取记忆变量添加到提示词中
# prompt = '{chat_story}, {query}'
# input =
"""
content = chain.invoke({"query":input, "chat_history":memory_var.get("content")})
memory.save_context({"query":input, "output":content})
"""

