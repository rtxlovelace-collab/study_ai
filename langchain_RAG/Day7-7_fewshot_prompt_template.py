import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
#少样示例模板
example = [{"question":"帮我计算2*2等于多少？", "answer":"4"},
            {"question":"帮我计算2+2等于多少？", "answer":"4"},
           {"question":"帮我计算12*12等于多少？", "answer":"144"},
           ]
example_prompt = ChatPromptTemplate.from_messages([
    ("human","{question}"),
    ("ai","{answer}")
])
#构建少量示例提示词模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples=example,
)
'''FewShotChatMessagePromptTemplate（少样本对话消息提示模板）是大语言模型（LLM）提示工程中用于 
 少样本学习（Few-Shot Learning） 的对话式提示模板，
核心作用是通过 “提供少量示例”，让模型快速理解任务目标、输出格式或语境要求，尤其适用于对话场景下的任务引导'''
print("少量提示词示例模板：", few_shot_prompt.format())
#构建最终的提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个可以计算复杂数学问题的聊天机器人，结果直接告诉我。"),
    few_shot_prompt,
    ("human", "{question}")
])
#创建大语言模型与链
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt|llm|StrOutputParser()

# 5.调用链获取结果
print(chain.invoke("帮我计算下14*15等于多少"))