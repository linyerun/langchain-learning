from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from rich import print as rprint


load_dotenv()
base_url = os.getenv('DASHSCOPE_BASE_URL')
api_key = os.getenv('DASHSCOPE_API_KEY')


model = ChatOpenAI(
    model='qwen3.8-max',
    base_url=base_url,
    api_key=api_key,
)


# 2.使用Pydantic结构化方式定义
class ContractInfo(BaseModel):
    """用户的联系方式"""
    name : str = Field(description="用户的姓名")
    email : str = Field(description="用户的邮箱")
    phone : str = Field(description="用户的电话")


agent = create_agent(
    model = model,
    response_format=ProviderStrategy(ContractInfo)
)


response = agent.invoke({
    "messages": [
        {"role":"user","content":"从以下信息中提取用户信息，小明的邮箱是shkstart@atguigu.com,电话是13012341234"}
    ]
})


rprint(response)
