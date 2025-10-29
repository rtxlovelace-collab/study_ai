from langchain.chains import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import Any, Dict, List, Optional


# 假设的LLM客户端类（实际使用时替换为真实的LLM）
class MockLLMClient:
    def chat(self, messages: List[Dict[str, str]]) -> str:
        # 模拟LLM生成回复
        return f"模拟回复: {messages[-1]['content']}"


class CustomChatMemory(BaseChatMemory):
    def __init__(self, max_token_limit: int = 1000, client: MockLLMClient = None):
        super().__init__(return_messages=False)  # 与实现保持一致
        self.chat_histories: List[Dict[str, str]] = []  # 存储对话历史
        self.max_token_limit = max_token_limit  # 修正变量名
        self.summary = ""  # 修正属性名
        self._client = client or MockLLMClient()  # 初始化客户端

    @property
    def memory_variables(self) -> List[str]:
        """返回内存变量名列表"""
        return ["chat_histories"]

    def _get_buffer_string(self) -> str:
        """正确拼接所有对话历史"""
        buffer = ""
        for chat in self.chat_histories:
            buffer += f"Human: {chat.get('human')}\nAI: {chat.get('ai')}\n"
        return buffer.strip()  # 移到循环外，返回所有记录

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载内存变量（符合父类参数要求）"""
        buffer_string = self._get_buffer_string()

        # 如果历史过长，生成摘要
        if self.get_num_tokens(buffer_string) > self.max_token_limit:
            self.summary = self.summary_text(buffer_string)
            return {"chat_histories": f"摘要：{self.summary}\n历史信息：{buffer_string}"}
        return {"chat_histories": buffer_string}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """保存对话上下文（符合父类参数要求）"""
        human_query = inputs.get("query", "")
        ai_response = outputs.get("text", "")

        self.chat_histories.append({
            "human": human_query,
            "ai": ai_response
        })

    def summary_text(self, buffer: str) -> str:
        """生成对话摘要"""
        if not self._client:
            return "摘要生成失败：缺少LLM客户端"

        messages = [
            {"role": "system", "content": "请简要总结以下对话内容"},
            {"role": "user", "content": buffer}
        ]
        return self._client.chat(messages)

    def get_num_tokens(self, text: str) -> int:
        """简单的token计数（实际应使用模型的tokenizer）"""
        return len(text) // 4  # 粗略估算：1 token ≈ 4字符


# 主程序
def main():
    # 初始化组件
    llm_client = MockLLMClient()
    memory = CustomChatMemory(max_token_limit=200, client=llm_client)

    # 构建提示模板（使用正确的内存变量名）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个聊天机器人，请根据历史对话和当前问题进行回复。"),
        MessagesPlaceholder(variable_name="chat_histories"),  # 匹配内存返回的键名
        ("human", "{query}")
    ])

    # 创建对话链
    chain = LLMChain(
        llm=llm_client,  # 实际使用时替换为真实LLM
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    # 对话循环
    print("开始聊天（输入'退出'结束）：")
    while True:
        query = input("你: ")
        if query.lower() == "退出":
            break

        # 调用链生成回复（使用正确的变量名）
        result = chain.invoke({"query": query})
        ai_reply = result["text"]
        print(f"AI: {ai_reply}")


if __name__ == "__main__":
    main()
