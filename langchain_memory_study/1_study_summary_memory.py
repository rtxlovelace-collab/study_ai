'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 1_study_summary_memory.py
* @Time: 2025/9/1
* @All Rights Reserve By Brtc
'''
from typing import Any
import dotenv
from openai import OpenAI


dotenv.load_dotenv()
# 记忆功能
# 1、max_tokens 用于传递最大的缓冲记忆令牌数
# 2、summary 用于存储记忆摘要
# 3、chat_hostories用于存储历史会话
# 4、get_nums_tokens 用于计算文本 tokens数量
# 5、save_context 用于存储新的对话信息
# 6、get_buffer_string 用于将历史对话转换成字符串
# 7、load_memory_variables用于加载记忆信息
# 8、summary_text用于生成摘要

#作业：
""" 
1、读一个问txt
2、把文件内容返回
3、整体功能 就是根据 txt文件来回答 txt 文件 问题
"""
class ConversationSummaryBufferMemory:
    def __init__(self, summary:str="", chat_histories:list=None, max_tokens:int = 300):
        """初始化函数"""
        self._summary = summary
        self._chat_histories = [] if chat_histories is None else chat_histories
        self._max_tokens = max_tokens
        self._client = OpenAI(base_url="https://api.ephone.chat/v1")
    def summary_text(self, origin_summary:str, new_chat:str):
        """根据老的摘要 + 新的对话 生成新的摘要"""
        prompt = f"""  
        你是一个强大的聊天机器人，请根据用户提供的谈话内容，总结摘要，并将其添加到先前提供的摘要中，返回一个新的摘要，除了新摘要其他任何数据都不要生成，如果用户的对话信息里有一些关键的信息，比方说姓名、爱好、性别、重要事件等等，这些全部都要包括在生成的摘要中，摘要尽可能要还原用户的对话记录。
        请不要将<example>标签里的数据当成实际的数据，这里的数据只是一个示例数据，告诉你该如何生成新摘要。
        <example>
        当前摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股向善的力量。
        新的对话：
        Human：为什么你认为人工智能是一股向善的力量？
        AI：因为人工智能会帮助人类充分发挥潜力。
        新摘要：人类会问人工智能对人工智能的看法，人工智能认为人工智能是一股向善的力量，因为它将帮助人类充分发挥潜力。
        </example>
        =====================以下的数据是实际需要处理的数据=====================
        当前摘要：{origin_summary}
        新的对话：
        {new_chat}
        请帮用户将上面的信息生成新摘要。
        """
        completion = self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user", "content":prompt}],
        )
        return completion.choices[0].message.content
    def get_buffer_string(self)->str:
        """返回历史字符串消息"""
        buffer = ""
        for chat in self._chat_histories:
            buffer += f"Human:{chat.get('human')}\n AI:{chat.get('ai')}\n\n"
        return buffer.strip()
    def load_memory_variables(self)->dict[str, Any]:
        """返回携带历史信息字典"""
        buffer_string = self.get_buffer_string()
        return{"chat_histories":f"摘要:{self._summary}\n\n 历史信息:{buffer_string}"}
    def save_context(self, human_query, ai_content:str)-> None:
        """保存一条全新的聊天记录"""
        self._chat_histories.append({"human": human_query, "ai": ai_content})
        buffer_string = self.get_buffer_string()
        tokens = self.get_num_tokens(buffer_string)
        if tokens > self._max_tokens:
            frist_chat = self._chat_histories[0]
            print("==============================新摘要生成中.......")
            self._summary = self.summary_text(self._summary,
                                              f"Human:{frist_chat.get('human')}\n AI:{frist_chat.get('ai')}\n\n")
            print("===============================摘要生成完成！")
            del self._chat_histories[0]

    @classmethod
    def get_num_tokens(cls, query:str)->int:
        """获取文本的token数量 """
        return len(query)


client = OpenAI(base_url="https://api.ephone.chat/v1")
if __name__ == "__main__":
    app_mem = ConversationSummaryBufferMemory("", [], 300)
    while True:
        human_input = input("Human:")
        if human_input == "exit":
            print("Goodbye")
            exit(0)
        memory_variables = app_mem.load_memory_variables()
        answer_prompt = (
            "你是一个强大的聊天机器人， 请根据对应的上下文和用户的提问，来回答用户的问题.\n\n"
            f"{memory_variables.get('chat_histories')}"
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
        app_mem.save_context(human_input, ai_content)