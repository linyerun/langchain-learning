from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
# 都会强行用 .env 文件里写的值去覆盖它
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 创建LLM
deepseek_llm = init_chat_model(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-flash"
)


@tool
def get_weather(city: str):
    """获取天气的工具"""
    return f"{city}天气晴朗~"


# 将模型和工具绑定
model_with_tools = deepseek_llm.bind_tools([get_weather])

# 声明一个消息列表
messages = [
    HumanMessage("今天北京天气如何")
]

# 模型生成调用工具请求
response = model_with_tools.invoke(messages)
messages.append(response)  # 直接append AIMessage，不用手动new AIMessage

# 看是否有方法调用
tool_calls = response.tool_calls
for tool_call in tool_calls:
    if tool_call.get('name', '') == "get_weather":
        tool_response = get_weather.invoke(tool_call)
        messages.append(ToolMessage(
            content=tool_response,
            tool_call_id=tool_call['id']
        ))

# 打印messages里面最新的消息
print("=====================> messages <=====================")
for msg in messages:
    msg.pretty_print()

# 重新调用模型
print("=====================> final response <=====================")
final_response = model_with_tools.invoke(messages)
print(f"final_response: \n{final_response.content}")
