from langgraph.checkpoint.postgres import PostgresSaver
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage

# 基于pg的短期记忆

# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
# 都会强行用 .env 文件里写的值去覆盖它
load_dotenv(override=True)


# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")


# 创建LLM
model = init_chat_model(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek:deepseek-flash" # 冒号前面直定模型厂商，冒号后面指定模型
)

DB_URL = "postgresql://langchain_user:abcd1234@127.0.0.1:5432/langchain_db?sslmode=disable"


with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    # 初始化postgre数据库
    checkpointer.setup()

    agent = create_agent(
        model=model,
        checkpointer=checkpointer
    )

    config = {
        "configurable" : {
            "thread_id" : "1001"
        }
    }

    response1 = agent.invoke({
        "messages": [HumanMessage("你好，我是林叶润")]},
        config=config
    )
    for msg in response1['messages']:
        msg.pretty_print()


    response2 = agent.invoke({
        "messages": [HumanMessage("你知道我是谁吗？")]},
        config=config
    )
    for msg in response2['messages']:
        msg.pretty_print()
