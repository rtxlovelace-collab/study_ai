'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 2_study_file_memeory.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
import dotenv
from openai import OpenAI
from langchain_community.chat_message_histories import FileChatMessageHistory
dotenv.load_dotenv()
chat_history = FileChatMessageHistory("./chat_mem.txt")
client = OpenAI(base_url="https://api.ephone.chat/v1")
if __name__ == "__main__":
    while True:
        human_input = input("Human:")
        if human_input == "exit":
            print("Goodbye")
            exit(0)
        answer_prompt = (
            "你是一个强大的聊天机器人， 请根据对应的上下文和用户的提问，来回答用户的问题.\n\n"
            f"<context>{chat_history}</context>"
            f"用户的提问是:{human_input}"
        )
        print("AI:", end="", flush=True)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是OpenAI开发的机器人，请根据 要求回答问题！"},
                {"role": "user", "content": answer_prompt},
            ],
            stream=True,
        )
        ai_content = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is None:
                break
            ai_content += content
            print(content, flush=True, end="")
        print("")
        chat_history.add_user_message(human_input)
        chat_history.add_ai_message(ai_content)