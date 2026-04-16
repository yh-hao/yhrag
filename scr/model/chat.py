from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseChatModel:
    def __init__(self, model_name: str="llama3.1:8b", system_prompt: str="你是一个乐于助人的助手。", retriever=None):
        self.model_name = model_name
        self.chat_model = ChatOllama(model=self.model_name, keep_alive=-1)
        self.retriever = retriever
        self.system_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.base_chain = self.system_prompt | self.chat_model
        self.store = {}

        # 5. 包装成带历史的 Chain

        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\n 请参考以下【检索到的上下文】来回答问题：\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        if self.retriever:
            # RAG 链：输入 -> 检索 -> 注入上下文 -> 模型生成
            self.base_chain = (
                    {
                        "context": self.retriever,  # 自动调用 retriever 获取文档
                        "input": RunnablePassthrough(),  # 原样传递用户输入
                        "chat_history": RunnablePassthrough()  # 原样传递历史
                    }
                    | self.rag_prompt
                    | self.chat_model
                    | StrOutputParser()  # 将模型输出转为字符串
            )
        else:
            # 保持兼容性：如果没有检索器，就用原来的纯对话模式
            legacy_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            self.base_chain = legacy_prompt | self.chat_model | StrOutputParser()

            # 4. 包装成带历史的 Chain
            # 注意：这里的 RunnableWithMessageHistory 包装的是重构后的 base_chain
        self.chat_chain = RunnableWithMessageHistory(
            self.base_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def get_session_history(self, session_id: str):
        """
        获取指定会话的历史记录，如果不存在则创建新的
        """
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def chat(self, user_input, session_id="default_user"):
        """
        发送消息并获取回复
        :param user_input: 用户输入
        :param session_id: 会话ID，用于区分不同用户或对话窗口
        """
        config = {"configurable": {"session_id": session_id}}
        response = self.chat_chain.invoke({"input": user_input}, config=config)
        return response.content

    def stream_chat(self, user_input, session_id="default_user"):
        """
        流式发送消息并获取回复
        :param user_input: 用户输入
        :param session_id: 会话ID，用于区分不同用户或对话窗口
        """
        config = {"configurable": {"session_id": session_id}}
        response = self.chat_chain.stream({"input": user_input}, config=config)
        for chunk in response:
            yield chunk.content

    def clear_history(self, session_id="default_user"):
        """
        清空指定会话的历史记录
        """
        if session_id in self.store:
            self.store[session_id].clear()
            print(f"会话 {session_id} 的历史已清空。")


if __name__ == "__main__":
    # 实例化机器人
    my_bot = BaseChatModel()

    # 模拟多轮对话
    print("🤖 机器人: 你好！我是你的助手。")

    # 第一轮
    response1 = my_bot.chat("我叫什么", session_id="user_001")
    print(f"🤖 机器人: {response1}")




