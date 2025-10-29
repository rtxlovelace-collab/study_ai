from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import ConfigurableField

prompt = PromptTemplate.from_template("写一篇关于{subject}的故事").configurable_fields(
    template = ConfigurableField(id = "llmprompt_template",name = "提示词模板",description="模板")

)
#传递配置更改llmprompt_template并调用生成内容
#动态更改提示词模板
content= prompt.invoke({"subject":"程序员"},
                       config={"configurable":{"llmprompt_template": "写一篇关于{subject}的笑话"}}
                       )

print(content.to_string())