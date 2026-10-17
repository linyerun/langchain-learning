from langchain.agents import create_agent
from langgraph.config import get_stream_writer
from langchain.tools import tool
import time
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

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
def generate_sales_report() -> str:
    """生成销售报告"""

    # 模拟数据处理
    writer = get_stream_writer()
    writer({"type": "生成销售报告", "message": "开始生成销售报告"})
    for i in range(1, 4):
        time.sleep(0.5)
        writer({"type": "生成销售报告","message": f"生成销售报告进度百分比：{i * 25}%"})
    writer({"type": "生成销售报告", "message": "报告生成完成"})

    return f"销售报告：总收入150万元，同比增长12%"


@tool
def generate_inventory_report() -> str:
    """生成库存报告"""

    # 模拟数据处理
    writer = get_stream_writer()
    writer("开始库存分析...")
    time.sleep(0.5)
    writer("检查当前库存量...")
    time.sleep(0.5)
    writer("生成库存报告...")

    return "当前库存量为10000件，库存充足，无异常"


# 创建报告生成agent
reporting_agent = create_agent(
    model=model,
    tools=[generate_sales_report, generate_inventory_report]
)

for chunk in reporting_agent.stream(
        {"messages": [{"role": "user","content": "生成销售报告和库存报告"}]},
        stream_mode="custom"
):
    print(chunk)
    print("-" * 50)
