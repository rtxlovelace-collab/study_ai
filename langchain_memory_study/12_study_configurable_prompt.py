'''
* This is the projet for Brtc LlmOps Platform
* @Author Leon-liao <liaosiliang@alltman.com>
* @Description //TODO 
* @File: 12_study_configurable_prompt.py
* @Time: 2025/9/2
* @All Rights Reserve By Brtc
'''
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField

prompt = PromptTemplate.from_template("请写一篇关于{subject}主题的冷笑话").configurable_fields(
    template = ConfigurableField(id = "llmprompt_template", name = "提示词模板", description="模板")
)
#传递配置更改prompt_template并调用生成内容
content = prompt.invoke({"subject":"程序员"},
                        config={"configurable":{"llmprompt_template":"请写一篇关于{subject}的藏头诗"}}
                        )
print(content.to_string())
