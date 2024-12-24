from guidance import models, gen
import json
import re

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
        prompt = f"输入: {input_text}\n输出的情感标签是(喜悦, 悲伤, 生气, 恐惧, 惊讶, 厌恶):\n情感: "

        # 使用模型生成情感标签
        result = lm + prompt + gen(max_tokens=20)  # 控制生成最大 token 数量

        # 确保 result 是字符串
        if isinstance(result, str):
            result = result.strip()
        else:
            result = str(result)

        print("Generated text:")
        print(result)

        # 使用正则表达式提取模型生成的情感标签
        emotion_match = re.search(r'(喜悦|悲伤|生气|恐惧|惊讶|厌恶)', result)
        predicted_emotion = emotion_match.group(1) if emotion_match else "未识别情感"

        # 构建JSON响应
        response_json = {
            "input": input_text,
            "predicted_emotion": predicted_emotion,  # 返回生成的情感标签
            "expected_emotion": expected_emotion
        }
        print(json.dumps(response_json, ensure_ascii=False, indent=2))

        # 检查预测是否正确
        is_correct = predicted_emotion == expected_emotion
        return response_json, is_correct
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        print(f"Error type: {type(e)}")
        return None, False


# 模拟数据集
dataset = [
    {"input": "哈哈哈，好可爱啊", "expected_emotion": "喜悦"},
    {"input": "这太让人失望了", "expected_emotion": "悲伤"},
    {"input": "你怎么能这样做？太过分了", "expected_emotion": "生气"},
    {"input": "我终于成功了，真是太棒了！", "expected_emotion": "喜悦"},
    {"input": "他竟然背叛了我，我感到非常难过", "expected_emotion": "悲伤"},
    {"input": "你这样做简直不可理喻！", "expected_emotion": "生气"},
    {"input": "今天的天气真好，心情也跟着愉快起来", "expected_emotion": "喜悦"},
    {"input": "考试没考好，我感到很沮丧", "expected_emotion": "悲伤"},
    {"input": "你太过分了，我再也不会原谅你", "expected_emotion": "生气"},
    {"input": "看到孩子们的笑脸，我感到无比幸福", "expected_emotion": "喜悦"},
    {"input": "失去亲人的痛苦让我无法释怀", "expected_emotion": "悲伤"},
    {"input": "你这样做真是太不负责任了！", "expected_emotion": "生气"},
    {"input": "听到好消息，我忍不住笑了起来", "expected_emotion": "喜悦"},
    {"input": "遭遇挫折，我感到非常失落", "expected_emotion": "悲伤"},
    {"input": "你太过分了，我真的很生气", "expected_emotion": "生气"},
    {"input": "看到美丽的风景，心情变得非常愉快", "expected_emotion": "喜悦"},
    {"input": "失去工作让我感到非常沮丧", "expected_emotion": "悲伤"},
    {"input": "你这样做简直不可理喻，我真的很生气", "expected_emotion": "生气"},
    {"input": "听到朋友的鼓励，我感到非常开心", "expected_emotion": "喜悦"},
    {"input": "遭遇失败，我感到非常难过", "expected_emotion": "悲伤"},
    {"input": "你太过分了，我再也不会原谅你", "expected_emotion": "生气"}

]

# 使用 Qwen/Qwen2.5-0.5B 模型
model_name = "Qwen/Qwen2.5-0.5B"

# 评估模型
correct = 0
for sample in dataset:
    response, is_correct = run_model_json(model_name, sample["input"], sample["expected_emotion"])
    if is_correct:
        correct += 1

# 计算准确率
accuracy = correct / len(dataset) * 100
print(f"Accuracy: {accuracy:.2f}%")