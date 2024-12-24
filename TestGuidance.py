from guidance import models, gen, select
import json
import time


def run_model_json(model_name, input_text, expected_emotion, model_kwargs=None):
    print(f"\nAttempting to run model: {model_name}")
    try:
        # 设置默认参数
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs.setdefault('trust_remote_code', True)

        # 初始化模型
        lm = models.Transformers(model_name, **model_kwargs)
        print(f"Successfully initialized model {model_name}")

        # 构建用于生成情感的提示
        prompt = f"输入: {input_text}\n输出的情感标签是: "

        # 使用 guidance 受控生成，限制情感标签只能从给定选项中选择
        guidance_program = (
                prompt + select(["喜悦", "悲伤", "生气", "恐惧", "惊讶", "厌恶"], name="emotion")
        )

        # 记录开始时间
        start_time = time.time()

        # 使用模型生成情感标签
        result = lm + guidance_program

        # 记录结束时间
        end_time = time.time()
        latency = end_time - start_time

        # 获取模型生成的情感
        predicted_emotion = result["emotion"].strip()

        # 构建JSON响应
        response_json = {
            "input": input_text,
            "predicted_emotion": predicted_emotion,  # 返回生成的情感标签
            "expected_emotion": expected_emotion,
            "latency": f"{latency:.4f}s"  # 记录时延
        }
        print(json.dumps(response_json, ensure_ascii=False, indent=2))

        # 检查预测是否正确
        is_correct = predicted_emotion == expected_emotion
        return response_json, is_correct, latency
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        print(f"Error type: {type(e)}")
        return None, False, 0


# 示例代码：使用数据集测试模型
dataset = [
    {"input": "哈哈哈，好可爱啊", "expected_emotion": "喜悦"},
    {"input": "这太让人失望了", "expected_emotion": "悲伤"},
    {"input": "你怎么能这样做？太过分了", "expected_emotion": "生气"}
]

model_name = "phi3.5:3.8b"
# phi3.5:3.8bc
# Phi/Phi3.5-3.8b
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
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average Latency: {average_latency:.4f}s")