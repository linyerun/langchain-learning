from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from datetime import datetime

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

messagesList = [
    [{'role': 'user', 'content': '你好，你是谁？'}],
    [HumanMessage('2+2=?')],
    '中国首都在哪里？'
]

# case1
print("case1<==========")
start_t = datetime.now()
responses = deepseek_llm.batch(messagesList)
end_t = datetime.now()
print('case1 cost:',end_t - start_t)
for response in responses:
    print(response.content)
    print()
    print()
    print("==========")
    print()
    print()

# case2
print("case2<==========")
start_t = datetime.now()
responses = deepseek_llm.batch_as_completed(messagesList)
end_t = datetime.now()
print(end_t - start_t)
print('case2 cost:',end_t - start_t)
for response in responses:
    print(f'{response[0]}: {response[1].content}')
    print()
    print()
    print("==========")
    print()
    print()
