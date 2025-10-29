import dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from twisted.names.client import query

dotenv.load_dotenv()


from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_messages([
    ("system","根据用户的问题回答,<context>{context}</context>。"),
    ("human","{query}")
    ])
llm = ChatOpenAI(model="gpt-4o-mini")

chain = create_stuff_documents_chain(prompt=prompt, llm=llm)
'''，create_stuff_documents_chain 是一个用于构建文档处理链的核心工具函数，
核心作用是将 “用户查询” 与 “相关文档内容” 高效结合'''

#文档列表
documents = [
    Document(page_content="小明喜欢玩游戏，他最近在玩一个叫做《原神》的游戏。"),
    Document(page_content="小王最近在看一本书，书名叫做《如何阅读一本书》。"),
    Document(page_content="小红昨天天在看电影，电影名叫做《盗墓笔记》今天在和小明一起玩游戏")
    ]
content = chain.invoke({"query":"说他们都干了什么事情说","context":documents})
print(content)