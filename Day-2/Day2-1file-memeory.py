import os

from typing import Any

import dotenv
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from openai import OpenAI
from twisted.conch.ssh.connection import messages
from twisted.names.client import query

dotenv.load_dotenv()
chat_history = FileChatMessageHistory("/chat_mam.txt")
client = OpenAI(base_url='https://api.ephone.chat/v1')#chatgpt客户端

if __name__ == "__main__":

    while True:
        human_input = input("Human: ")
        if human_input == "q":
            print("Goodbye")
            exit(0)
        # with open("../作业/作业引用.txt","r",encoding="utf-8") as f:
        #     world_text = f.read()
        answer_prompt =(
            "你是一个很厉害的聊天机器人，请根据上下文给用户一个答案。"
            
            f"当前摘要：<context>{chat_history}</context>\n"
            f"用户的提问是：{human_input}"
        )
        print("AI:",end="",flush=True)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role" : "system","content" : "你是基于openai的聊天机器人，根据要求回答问题！"},
                {"role":"user","content":answer_prompt},
                    ],
                stream=True
        )
        ai_content = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is None:
                break
            ai_content += content
            print(content,end="",flush=True)
        print("")
        chat_history.add_user_message(human_input)
        chat_history.add_ai_message(ai_content)


