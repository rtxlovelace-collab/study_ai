#做一个有记忆的ai聊天机器人，
# 可以记住与用户之间的对话，
# 并根据对话生成相应的回复。
# 记忆内容放在向量数据库中
# 每次对话之前都要检索记忆
#每次的对话都要放在向量数据库中，并与之前的对话进行对比，生成相似度，并根据相似度生成回复。
import os
import openai
import weaviate
from langchain_openai import OpenAIEmbeddings
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Property, DataType
import time
from dotenv import load_dotenv
from weaviate.auth import AuthApiKey
# 加载环境变量
load_dotenv()

# 配置API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")


# 连接到Weaviate云服务
def connect_weaviate_cloud():
    """连接到Weaviate云服务并返回客户端"""
    try:
        # 从环境变量获取Weaviate云服务信息
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_KEY")

        if not weaviate_url or not weaviate_api_key:
            print("请确保环境变量中设置了WEAVIATE_CLOUD_URL和WEAVIATE_API_KEY")
            exit(1)

        # 配置Weaviate客户端
        client = weaviate.connect_to_local(os.getenv("WEAVIATE_HOST"))
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        # client = weaviate.connect_to_weaviate_cloud(
        #     url=weaviate_url,
        #     auth_client_secret=weaviate.AuthApiKey(api_key=weaviate_api_key),
        #     additional_headers={
        #         "X-OpenAI-Api-Key": openai.api_key  # 用于Weaviate中的OpenAI模块
        #     }
        # )

        # 检查连接
        client.schema.get()
        print("成功连接到Weaviate云服务")
        return client
    except Exception as e:
        print(f"连接Weaviate云服务失败: {e}")
        print("请检查你的Weaviate云服务URL和API密钥是否正确")
        exit(1)


# 创建对话记忆集合
def create_chat_memory_class(client):
    """创建用于存储对话记忆的Weaviate集合"""
    class_name = "ChatMemory"

    # 检查集合是否已存在，不存在则创建
    if not client.schema.exists(class_name):
        # 定义集合结构
        client.schema.create_class({
            "class": class_name,
            "properties": [
                {
                    "name": "role",
                    "dataType": ["string"],
                    "description": "角色，user或assistant"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "对话内容"
                },
                {
                    "name": "timestamp",
                    "dataType": ["number"],
                    "description": "时间戳"
                }
            ],
            "vectorizer": "embedding",  # 使用Weaviate云服务中的OpenAI向量器
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "ada",
                    "modelVersion": "002",
                    "type": "text"
                }
            }
        })
        print(f"已创建 {class_name} 集合")
    else:
        print(f"{class_name} 集合已存在")


# 存储对话到Weaviate云服务
def store_conversation(client, role, content):
    """将对话内容存储到Weaviate云服务（自动处理embedding）"""
    class_name = "ChatMemory"

    # 存储到Weaviate，使用云服务内置的vectorizer
    client.data_object.create(
        data_object={
            "role": role,
            "content": content,
            "timestamp": time.time()
        },
        class_name=class_name
    )
    print(f"已存储{role}的对话到Weaviate云服务")


# 检索相关的历史对话
def retrieve_relevant_conversations(client, query, limit=5, similarity_threshold=0.7):
    """从Weaviate云服务检索与当前查询相关的历史对话"""
    class_name = "ChatMemory"

    # 执行向量搜索
    response = client.query.get(
        class_name,
        ["role", "content", "timestamp"]
    ).with_near_text({"concepts": [query]}).with_limit(limit).do()

    # 处理结果
    relevant_conversations = []
    if "data" in response and "Get" in response["data"] and class_name in response["data"]["Get"]:
        for item in response["data"]["Get"][class_name]:
            # 提取相似度分数
            similarity = item.get("_additional", {}).get("certainty", 0)

            if similarity >= similarity_threshold:
                relevant_conversations.append({
                    "role": item["role"],
                    "content": item["content"],
                    "timestamp": item["timestamp"],
                    "similarity": similarity
                })

    # 按时间戳排序
    relevant_conversations.sort(key=lambda x: x["timestamp"])

    return relevant_conversations


# 生成回复
def generate_response(query, relevant_conversations):
    """根据当前查询和相关历史对话生成回复"""
    # 构建提示词
    prompt = "你是一个有记忆的聊天机器人。请根据当前问题和相关的历史对话，生成合适的回答。\n\n"

    if relevant_conversations:
        prompt += "相关的历史对话：\n"
        for conv in relevant_conversations:
            prompt += f"{conv['role']}: {conv['content']} (相似度: {conv['similarity']:.2f})\n"
        prompt += "\n"

    prompt += f"当前问题：{query}\n"
    prompt += "你的回答："

    # 调用OpenAI生成回复
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你是一个有记忆的聊天机器人，会根据历史对话上下文回答问题。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message['content']


# 聊天主函数
def chat_loop(client):
    """聊天机器人主循环"""
    print("欢迎使用带记忆功能的聊天机器人！输入 'exit' 退出聊天。")

    while True:
        # 获取用户输入
        user_input = input("你: ")

        if user_input.lower() == 'exit':
            print("再见！")
            break

        # 检索相关历史对话
        print("正在检索相关历史对话...")
        relevant_convs = retrieve_relevant_conversations(client, user_input)

        # 显示检索到的相关对话
        if relevant_convs:
            print(f"\n找到 {len(relevant_convs)} 条相关历史对话：")
            for i, conv in enumerate(relevant_convs, 1):
                print(f"{i}. {conv['role']}: {conv['content']} (相似度: {conv['similarity']:.2f})")
            print()

        # 生成回复
        print("正在生成回复...")
        response = generate_response(user_input, relevant_convs)

        # 显示回复
        print(f"AI: {response}\n")

        # 存储对话到Weaviate
        store_conversation(client, "user", user_input)
        store_conversation(client, "assistant", response)


# 主函数
def main():
    # 连接到Weaviate云服务
    client = connect_weaviate_cloud()

    # 创建对话记忆集合
    create_chat_memory_class(client)

    # 开始聊天
    chat_loop(client)


if __name__ == "__main__":
    main()
