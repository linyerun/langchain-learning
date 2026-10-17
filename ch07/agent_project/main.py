from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from smart_assistant import SmartAssistant

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

# ==================== 主程序 ====================

def main():
    assistant = SmartAssistant(model)

    print("=" * 40)
    print("🤖 多功能智能助手（LangChain 1.2）")
    print("=" * 40)
    print("\n我可以帮你：")
    print("  🌤️  查询天气")
    print("  🔢 数学计算")
    print("  ⏰ 时间查询")
    print("  💱 货币转换")
    print("  🔍 信息搜索")
    print("\n输入 'quit' 退出，输入 'reset' 重置对话\n")

    demos = [
        "北京今天天气怎么样？",
        "帮我算一下 (25 + 17) * 3",
        "现在几点了？",
        "100 美元等于多少人民币？"
    ]

    for demo in demos:
        print(f"👤 {demo}")
        response = assistant.chat(demo)
        print(f"🤖 {response}\n")

    # 重置对话
    assistant.reset()

    # 交互模式
    print("=" * 40)
    print("💬 进入交互模式")
    print("=" * 40)

    while True:
        user_input = input("\n👤 你: ")

        if user_input.lower() == 'quit':
            print("再见！👋")
            break

        if user_input.lower() == 'reset':
            assistant.reset()
            print("✅ 对话已重置")
            continue

        if not user_input.strip():
            continue

        # 调用助手
        response = assistant.chat(user_input)
        print(f"🤖 助手: {response}")

if __name__ == "__main__":
    main()
