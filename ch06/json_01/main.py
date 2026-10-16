from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv

# 通过load_dotenv()将.env中的变量加载为环境变量
# override=True表示：无论你当前的操作系统、终端或者虚拟环境中是否已经存在同名的环境变量，
# 都会强行用 .env 文件里写的值去覆盖它
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 创建DeepSeek LLM
model = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-flash"
)


"""
使用 JSON Schema 定义嵌套结构
"""
# 1. 定义嵌套的 JSON Schema
project_schema = {
    "title": "MovieInfo",
    "description": "包含电影标题、上映年份、导演、演员和评分的电影对象",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "电影标题"},
        "year": {"type": "integer", "description": "上映年份"},
        "director": {"type": "string", "description": "导演"},
        "cast": {  # 定义嵌套数组
            "type": "array",
            "description": "演员列表",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "演员姓名"},
                    "role": {"type": "string", "description": "演员角色"}
                },
                "required": ["name", "role"]
            }
        },
        "rating": {"type": "number", "description": "评分（10分制）"}
    },
    "required": ["title", "year", "director", "cast", "rating"]
}

# 绑定 JSON Schema 到模型
structured_model = model.with_structured_output(project_schema, method="function_calling", tool_choice='auto')

# 调用模型
response = structured_model.invoke("生成一个关于《星际穿越》的电影信息，包含导演、演员、评分")
print(response)

'''
{
    'title': '星际穿越', 
    'year': 2014, 
    'director': '克里斯托弗·诺兰', 
    'cast': [
        {'name': '马修·麦康纳', 'role': '库珀（Cooper）'}, 
        {'name': '安妮·海瑟薇', 'role': '布兰德（Brand）'}, 
        {'name': '杰西卡·查斯坦', 'role': '墨菲（Murph）'}, 
        {'name': '麦肯吉·弗依', 'role': '幼年墨菲'}, 
        {'name': '迈克尔·凯恩', 'role': '布兰德教授'}, 
        {'name': '马特·达蒙', 'role': '曼恩博士'},
    ], 
    'rating': 8.7,
}
'''