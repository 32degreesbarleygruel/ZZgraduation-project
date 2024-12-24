import subprocess
import json
import time
from  guidance import models,gen ,select

# 函数用于运行模型，并返回预测结果和时延
def run_model_json(model_name, input_text, expected_emotion):
    print(f"\n正在尝试运行模型: {model_name}")
    try:
        # 构建用于生成情感的提示
        # 构建用于生成情感的提示，只需要标签
        prompt = f"输入: {input_text}\n请只返回唯一答案，不要任何的多余文字,不要任何的符号，包括‘’,'',[]。"
        # prompt = f"输入: {input_text}\n请只返回情感标签：['喜悦', '悲伤', '生气', '恐惧', '惊讶', '厌恶'],不要任何的符号，包括‘’,'',[]。"

        # 然后使用 guidance 控制生成
        guidance_program = prompt + select(["喜悦", "悲伤", "生气", "恐惧", "惊讶", "厌恶"], name="emotion")

        # 这样可以让模型只返回你期望的情感标签
        # 记录开始时间
        start_time = time.time()

        # 使用subprocess运行ollama的命令
        process = subprocess.Popen(
            ["ollama", "run", model_name],
            stdin=subprocess.PIPE,  # 使用标准输入
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # 将提示内容传入标准输入
        stdout, stderr = process.communicate(input=prompt.encode())

        # 记录结束时间
        end_time = time.time()
        latency = end_time - start_time

        # 解析模型返回的输出
        output = stdout.decode().strip()
        if not output:
            raise Exception(f"模型运行错误: {stderr.decode().strip()}")

        # 打印错误输出（用于调试）
        # print(f"标准输出: {stdout.decode()}")
        # print(f"错误输出: {stderr.decode()}")

        # 假设输出格式为单个情感标签
        predicted_emotion = output.strip()
        result = subprocess.Popen(['ollama', 'run', model_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True,encoding='utf-8')
        output, error = result.communicate(input_text)

        # 构建JSON响应
        response_json = {
            "input": input_text,
            "predicted_emotion": predicted_emotion,
            "expected_emotion": expected_emotion,
            "latency": f"{latency:.4f}s"
        }
        print(json.dumps(response_json, ensure_ascii=False, indent=2))

        # 检查预测是否正确
        is_correct = predicted_emotion == expected_emotion
        return response_json, is_correct, latency

    except Exception as e:
        print(f"模型运行错误: {e}")
        return None, False, 0


# 示例数据集
dataset = [
    {"input": "2 + 2 = ?", "expected_output": "4"},
    {"input": "10 - 5 = ?", "expected_output": "5"},
    {"input": "3 * 3 = ?", "expected_output": "9"},
    {"input": "15 + 5 = ?", "expected_output": "20"},
    {"input": "20 / 4 = ?", "expected_output": "5"},
    {"input": "5 * 5 = ?", "expected_output": "25"},
    {"input": "30 + 10 = ?", "expected_output": "40"},
    {"input": "40 - 20 = ?", "expected_output": "20"},
    {"input": "6 * 6 = ?", "expected_output": "36"},
    {"input": "50 / 2 = ?", "expected_output": "25"},
    {"input": "7 * 7 = ?", "expected_output": "49"},
    {"input": "60 + 40 = ?", "expected_output": "100"},

]

# 选择模型名称
model_name = "phi3.5:3.8b"

# 评估模型
correct = 0
total_latency = 0
for sample in dataset:
    # 使用 `expected_output` 代替 `expected_emotion`
    response, is_correct, latency = run_model_json(model_name, sample["input"], sample["expected_output"])# expected_emotion
    if is_correct:
        correct += 1
    total_latency += latency

# 计算准确率和平均时延
accuracy = correct / len(dataset) * 100
average_latency = total_latency / len(dataset)
print(f"准确率: {accuracy:.2f}%")
print(f"平均时延: {average_latency:.4f}s")