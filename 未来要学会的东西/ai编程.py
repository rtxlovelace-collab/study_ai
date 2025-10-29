from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import dotenv
dotenv.load_dotenv()

def load_knowledge_base(docs_dir="./knowledge_docs"):
    """
    加载知识库文档并创建向量存储

    参数:
        docs_dir: './knowledge_docs/知识库.txt'

    返回:
        用于检索的向量存储检索器
    """
    # 检查知识库目录是否存在
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"已创建知识库目录: {docs_dir}，请将文档放入该目录")
        return None

    # 加载目录中的所有文本文件
    loader = DirectoryLoader(
        docs_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()

    if not documents:
        print(f"知识库目录 {docs_dir} 中未找到任何文档")
        return None

    # 将文档分割成小块
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    print(f"知识库加载完成，共分割为 {len(texts)} 个片段")

    # 创建嵌入并存储到向量数据库
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_documents(texts, embeddings)

    # 创建检索器，返回最相关的3个文档片段
    return db.as_retriever(search_kwargs={"k": 3})


def create_chatbot_with_knowledge(retriever, system_prompt=None, model_name="gpt-3.5-turbo", temperature=0.7):
    """
    创建具有上下文感知和知识库支持的聊天机器人

    参数:
        retriever: 知识库检索器
        system_prompt: 系统提示，定义机器人的行为和个性
        model_name: 使用的模型名称
        temperature: 模型生成的随机性

    返回:
        配置好的对话链
    """
    # 设置系统提示，如果未提供则使用默认提示
    if system_prompt is None:
        system_prompt = """你是知识库中的妹妹角色，请基于知识库中的设定对哥哥（用户）的话进行中文回答。
        同时你也是整个故事的驱动旁白，一起来玩一场文字冒险游戏吧！
        
        """

    # 定义提示模板
    prompt_template = system_prompt + """
    对话历史:
    {chat_history}

    上下文知识:
    {context}

    用户问题:
    {question}

    回答:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "context", "question"]
    )

    # 初始化内存以保存对话历史
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # 初始化聊天模型
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 创建结合检索和对话的链
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return conversation_chain


def chat_loop(conversation):
    """启动聊天循环"""
    print("欢迎使用带知识库的上下文感知聊天机器人！输入 'exit' 退出聊天。")

    while True:
        user_input = input("你: ")

        if user_input.lower() == 'exit':
            print("机器人: 再见！有任何问题随时来找我。")
            break

        # 获取机器人回应
        result = conversation({"question": user_input})
        print(f"机器人: {result['answer']}")

        # 显示参考的知识库来源（可选）
        if result['source_documents']:
            print("\n参考文档:")
            for i, doc in enumerate(result['source_documents'], 1):
                print(f"  {i}. {os.path.basename(doc.metadata['source'])}")
            print()


if __name__ == "__main__":
    # 加载知识库
    retriever = load_knowledge_base()

    # 如果没有加载到知识库，仍然可以作为普通聊天机器人运行
    if retriever is None:
        print("将以无知识库模式启动聊天机器人")
        # 创建没有知识库的聊天机器人（使用默认的ConversationChain）
        from langchain.chains import ConversationChain
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
            HumanMessagePromptTemplate

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        memory = ConversationBufferMemory(return_messages=True)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("你是一个博学多才的机器人，可以为你提供关于历史、文化、政治、哲学等方面的知识。"),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    else:
        # 创建带知识库的聊天机器人
        custom_system_prompt = """你是知识库中的妹妹角色，请基于知识库中的设定对哥哥（用户）的话进行中文回答。请优先使用提供的上下文知识回答问题，
        并结合对话历史保持回答的连贯性。如果无法从知识库中找到答案，请明确告知用户。
        回答要准确、简洁，用中文自然表达。
        """
        conversation = create_chatbot_with_knowledge(
            retriever=retriever,
            system_prompt=custom_system_prompt
        )

    # 启动聊天
    chat_loop(conversation)
