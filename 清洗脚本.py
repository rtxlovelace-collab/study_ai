import os
import re

import dotenv
from langchain import requests
from openai.types.conversations import text_content

dotenv.load_dotenv()

def fetch_data(query: str):
    """调用开放API获取对应Agent生成的文案内容"""
    # 1.从环境变量中获取数据
    url = os.getenv("LLMOPS_API_BASE")
    api_key = os.getenv("LLMOPS_API_KEY")
    app_id = os.getenv("LLMOPS_APP_ID")

    # 2.组装请求头
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 3.组装请求数据
    request_body = {
        "app_id": app_id,
        "end_user_id": "",
        "conversation_id": "",
        "stream": False,
        "query": query,
        "image_urls": [],
    }

    try:
        # 4.使用requests包发起post请求
        response = requests.post(url, json=request_body, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data["data"]["answer"]
        else:
            return f"请求失败: {response.status_code}"
    except Exception as e:
        return f"请求异常: {str(e)}"

def cut(text: str):
    """对文本进行分词"""
    return re.split(r'\n\n(?:0[1-9]|1[0-2])', text)

if __name__ == "__main__":
    with open("招聘数据源.txt", "r", encoding="utf-8") as f:
        text = f.read()
    data = cut(text)
    with open("清洗完毕.txt", "w", encoding="utf-8") as f_out:
        for i in range(1, len(data)):
        ai_result = fetch_data(data[i-1])
        print(ai_result)
        f_out.write(ai_result + "\n")  # 每行结果后加换行符，方便查看




