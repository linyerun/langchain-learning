from langchain_core.prompts import ChatPromptTemplate


chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个友好的AI助手，你的名字叫{name}"),
    ("human","你好，最近怎么样？"),
    ("ai","我很好，谢谢"),
    ("human","{user_input}")
])
messages = chat_prompt_template.invoke({"name":"小智", "user_input":"2 + 2 = ？"})
print(messages)
'''
messages=[SystemMessage(content='你是一个友好的AI助手，你的名字叫小智', additional_kwargs={}, response_metadata={}), HumanMessage(content='你好，最近怎么样？', additional_kwargs={}, response_metadata={}), AIMessage(content='我很好，谢谢', additional_kwargs={}, response_metadata={}, tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='2 + 2 = ？', additional_kwargs={}, response_metadata={})]
'''
