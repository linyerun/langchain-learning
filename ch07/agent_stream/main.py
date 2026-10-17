from langchain.agents import create_agent
from langchain.tools import tool
from typing import Dict, Any

from langchain_classic.agents.agent_toolkits import steam
from rich import print as rprint
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv


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


@tool
def query_customer_data(customer_id: str) -> Dict[str, Any]:
    """
    查询客户基本信息

    Args:
        customer_id: 客户ID，用于唯一标识客户

    Returns:
        包含客户基本信息的字典，如姓名、等级、加入日期等
    """
    # 模拟数据库查询
    return {"name": "张三","level": "VIP","join_date": "2023-01-15"}


@tool
def check_order_history(customer_id: str) -> Dict[str, Any]:
    """
    查询客户订单历史

    Args:
        customer_id: 客户ID，用于唯一标识客户

    Returns:
        包含客户订单历史的字典，如总订单数、总花费等
    """
    return {"total_orders": 15,"total_spent": 25800.00}


@tool
def get_current_promotions() -> Dict[str, Any]:
    """
    获取当前可用促销活动

    Returns:
        包含当前可用促销活动的字典，如活动名称、有效日期等
    """
    return {
        "promotions": ["老用户优惠", "会员专属折扣"],
        "valid_until": "2027-01-31"
    }


'''
# 创建客户服务Agent
customer_service_agent = create_agent(
    model=model,
    tools=[query_customer_data, check_order_history, get_current_promotions]
)

st = customer_service_agent.stream(
    {
        "messages" : [
            {"role":"user","content":"查询客户id为cust1234的完整的信息、历史订单和可用优惠"}
        ]
    },
    # stream_mode="values"
    # stream_mode="updates"
    # stream_mode="messages"
    # stream_mode="tasks"
    # stream_mode="debug"
)

for chunk in st:
    rprint(chunk)
    print("-" * 50)

# 打印messages类型的stream
# prev_node = ''
# prev_step = -1
# for chunk in st:
#     if chunk[1]['langgraph_node'] != prev_node and chunk[1]['langgraph_step'] != prev_step:
#         prev_node = chunk[1]['langgraph_node']
#         prev_step = chunk[1]['langgraph_step']
#         print()
#         print("-" * 50)
#
#     print(chunk[0].content, end='')
'''

from langgraph.checkpoint.memory import InMemorySaver

# 创建内存检查点存储
checkpointer = InMemorySaver()

# 创建agent
customer_service_agent = create_agent(
    model=model,
    tools=[query_customer_data, check_order_history, get_current_promotions],
    checkpointer=checkpointer# 启用检查点
)

# 3. 创建唯一的会话ID
config = {"configurable": {"thread_id": "session01"}}

# 4. 调用Agent
checkpoint_count = 0

# 使用checkpoints模式进行流式监控
for chunk in customer_service_agent.stream(
        {"messages": [{"role": "user","content": "查询客户ID为 CUST123456 的完整信息和可用优惠"}]},
        config=config,
        stream_mode="checkpoints"
):
    checkpoint_count += 1
    print(f"检查点 #{checkpoint_count}")
    print(chunk)
    print("-" * 50)
