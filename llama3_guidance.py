from guidance import models, gen
import transformers

model_name = "Llama3"
def run_model(model_name, prompt, model_kwargs=None):
    print(f"\nAttempting to run model: {model_name}")
    try:
        if model_kwargs is None:
            model_kwargs = {}
        model_kwargs.setdefault('trust_remote_code', True)

        lm = models.Transformers(model_name, **model_kwargs)
        print(f"Successfully initialized model {model_name}")

        result = lm + prompt + gen(max_tokens=50)
        print("Generated text:")
        print(result)
    except Exception as e:
        print(f"Error with {model_name}: {e}")
        print(f"Error type: {type(e)}")


# 使用 Llama3 模型
model_name = "Llama3"

# 1. 基本文本生成
run_model(model_name, "从前有座山，山上有座庙，庙里有个")

# 2. 问答示例
run_model(model_name, "Q: 什么是人工智能？\nA:")

# 3. 翻译示例
run_model(model_name, "将以下中文翻译成英文：我喜欢吃苹果。\n翻译:")

# 4. 多轮对话示例
try:
    chat_model = models.Transformers(model_name, trust_remote_code=True)
    conversation = chat_model + "Human: 你好，请问你是谁？\n" + "Assistant:" + gen(max_tokens=50)
    conversation += "\nHuman: 你能做些什么？\n" + "Assistant:" + gen(max_tokens=50)
    print("\n多轮对话:")
    print(conversation)
except Exception as e:
    print(f"多轮对话出错: {e}")
    print(f"错误类型: {type(e)}")