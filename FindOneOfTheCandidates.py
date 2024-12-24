import ollama

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
client = Client("qwen2.5:0.5b")

# 定义提示词c
prompt = """
# 内容
{"今天好⾟苦啊"}

# 任务
判断上述内容的情感。

# 要求
回答应仅包含识别结果（范围在 [喜悦|愤怒|厌恶|低落] 之间）不要掺杂其他任何话和符号
"""

# 发送请求并获取响应
response = client.chat(prompt)

# 输出结果
print(response)