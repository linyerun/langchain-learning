from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain.chat_models import init_chat_model

######1、提供大模型#########
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 创建LLM
deepseek_llm = init_chat_model(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek:deepseek-flash" # 冒号前面直定模型厂商，冒号后面指定模型
)


######2、提供提示词模板#########
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个友好的AI助手，你的名字叫{name}"),
    ("human","你好，最近怎么样？"),
    ("ai","我很好，谢谢"),
    ("human","{user_input}")
])

# 调用
prompt_value = chat_prompt_template.invoke({"name":"小智","user_input":"2 + 2 = ？"})

######3、模型调用#########
response = deepseek_llm.invoke(prompt_value)
print(response)
