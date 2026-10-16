from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from rich import print as rprint

# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
# 都会强行用 .env 文件里写的值去覆盖它
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 创建DeepSeek LLM
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-flash"
)

resp = deepseek_llm.invoke([
    {"role": "user", "content": "2 + 3 * 2 = ？"}
])


rprint(resp)
