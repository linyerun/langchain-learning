from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 1. 创建提示词模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system","回答用户问题,必须始终输出一个包含title(电影标题)和year(上映年份)的 JSON 对象"),
    ("human","问题：{question}")
])

# 2. 模型初始化
# 从.env文件中加载环境变量
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 创建DeepSeek LLM
model = init_chat_model(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek:deepseek-flash" # 冒号前面直定模型厂商，冒号后面指定模型
)

# 3. 定义结构
class Movie(BaseModel):
    """电影信息"""
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份")

# 4. 创建输出解析器
parser = JsonOutputParser(pydantic_object=Movie)

# 5. 创建链
chain = prompt_template | model | parser

# 6. 调用（返回字典）
response = chain.invoke({"question": "介绍电影《盗梦空间》"})
# response = parser.invoke(model.invoke(prompt_template.invoke({"question": "介绍电影《盗梦空间》"})))


print(type(response))
print(response)