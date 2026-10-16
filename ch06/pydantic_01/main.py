from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv

# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
# 都会强行用 .env 文件里写的值去覆盖它
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 创建DeepSeek LLM
model = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-flash"
)

from pydantic import BaseModel,Field
class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age : int = Field(description="年龄")
    occupation: str = Field(description="职业")

# 创建结构化输出的大语言模型
structured_model = model.with_structured_output(Person, tool_choice="auto")

# 调用模型，打印结果
result = structured_model.invoke("张三是一名30岁的软件工程师")
print(result)
print(type(result))
