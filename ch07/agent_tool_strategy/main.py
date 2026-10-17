from pydantic import BaseModel, Field
from langchain.agents.structured_output import ToolStrategy
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from rich import print as rprint
from dotenv import load_dotenv
import os
from langchain_deepseek import ChatDeepSeek
from typing import List, Optional, Any
from langchain_core.outputs import ChatResult


class SafeDeepSeek(ChatDeepSeek):
    """自动剥离 tool_choice 的 ChatDeepSeek 包装器，适配推理模式。"""

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 强制移除 tool_choice，避免 DeepSeek 推理模式报 400
        kwargs.pop("tool_choice", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


load_dotenv(override=True)


# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")


# 创建DeepSeek LLM
model = SafeDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-flash",
)


class ContactInfo(BaseModel):
    """用户的联系方式"""
    name: str = Field(description="用户姓名")
    email: str = Field(description="用户邮箱地址")
    phone: str = Field(description="用户的手机号")


agent = create_agent(
    model=model,
    response_format=ToolStrategy(schema=ContactInfo),
)


response = agent.invoke({
    "messages" : [
        HumanMessage(content="从这段话中抽取结构化信息：小明的邮箱是shkstart@atguigu.com,电话是：13012341234")
    ]
})


rprint(response)
