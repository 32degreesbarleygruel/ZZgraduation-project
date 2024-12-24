import time
import json
import subprocess
import guidance  # Make sure guidance is installed

# 使用 Guidance API 创建生成程序
guidance_program = guidance("""输入: {{input_text}}
请只返回情感标签：["喜悦", "悲伤", "生气", "恐惧", "惊讶", "厌恶"]，不要任何符号。""")

# 函数用于运行模型，并返回预测结果和时延
def run_model_json(model_name, input_text, expected_emotion):
    print(f"\n正在尝试运行模型: {model_name}")
    try:
        # 记录开始时间
        start_time = time.time()

        # 使用 guidance 运行模型并获取输出
        response = guidance_program(input_text=input_text)
        predicted_emotion = response['emotion']

        # 记录结束时间
        end_time = time.time()
        latency = end_time - start_time

        # 构建 JSON 响应
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
    {"input": "哈哈哈，好可爱啊", "expected_emotion": "喜悦"},
    {"input": "这太让人失望了", "expected_emotion": "悲伤"},
    {"input": "你怎么能这样做？太过分了", "expected_emotion": "生气"}
]

# 选择模型名称
model_name = "phi3.5:3.8b"

# 评估模型
correct = 0
total_latency = 0
for sample in dataset:
    response, is_correct, latency = run_model_json(model_name, sample["input"], sample["expected_emotion"])
    if is_correct:
        correct += 1
    total_latency += latency

# 计算准确率和平均时延
accuracy = correct / len(dataset) * 100
average_latency = total_latency / len(dataset)
print(f"准确率: {accuracy:.2f}%")
print(f"平均时延: {average_latency:.4f}s")