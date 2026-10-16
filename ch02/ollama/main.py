from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model


ollama_llm = ChatOllama(
    model="qwen3:0.6b",
    base_url="http://127.0.0.1:11434" #如果Ollama在本地默认端口运行则可省略
)
question = "你好，请你介绍一下你自己。"
result = ollama_llm.invoke(question)
print(result.content)

print("==========")

ollama_llm =init_chat_model(model="qwen3:0.6b", model_provider="ollama", base_url="http://127.0.0.1:11434")
print(ollama_llm.invoke(question).content)
