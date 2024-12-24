from enum import Enum
from pydantic import BaseModel, constr
import json

# 定义情感标签
class EmotionLabel(str, Enum):
    label1 = "喜悦"
    label2 = "愤怒"
    label3 = "厌恶"
    label4 = "低落"

# 定义情感模型
class Emotion(BaseModel):
    label: EmotionLabel
    reason: constr(max_length=200)

# 获取 Emotion 模型的 JSON Schema
extra_body = {
    "guided_json": Emotion.model_json_schema(),
    "guided_whitespace_pattern": ""
}

# 打印 JSON Schema 结构
print(Emotion.model_json_schema())

# 模拟输入的响应
response = '''
{
    "label": "愤怒",
    "reason": "使用了粗俗的语言，表达了愤怒情感。"
}
'''

# 解析并格式化 JSON 响应
try:
    data = json.loads(response)
    formatted_json = json.dumps(data, ensure_ascii=False, indent=4)
    print("Formatted JSON response:")
    print(formatted_json)
except json.JSONDecodeError:
    print("The response is not a valid JSON string. Here's the raw output:")
    print(response)