import ollama
import re

# 定义聊天客户端
class Client:
    def __init__(self, model_name):
        self.model_name = model_name

    def chat(self, prompt):
        llm_response = ollama.chat(
            model=self.model_name,
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0
            }
        )
        return llm_response['message']['content']

# 使用 Llama3 模型
client = Client("Llama3")

# 定义提示词
prompt = """
# 内容
哈哈哈哈哈

# 任务
判断上述内容的情感，回答应包含以下格式：
结果: [喜悦|愤怒|厌恶|低落] / 原因: [简短描述]。
"""

# 发送请求并获取响应
response = client.chat(prompt)

# 输出响应
print("原始响应:", response)

# 修改正则表达式，匹配中文符号和空格
pattern = r"结果[:：]\s*(喜悦|愤怒|厌恶|低落)\s*/\s*原因[:：]\s*[^\n]+"
match = re.search(pattern, response)

if match:
    result = match.group(0)
    print("", result)
else:
    print("未找到匹配内容")