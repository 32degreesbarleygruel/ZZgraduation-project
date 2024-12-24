import random
from faker import Faker
import json
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import ollama

fake = Faker()


# 随机生成 JSON schema 属性
def generate_random_property():
    property_types = [
        ('string', {'type': 'string'}),
        ('integer', {'type': 'integer'}),
        ('number', {'type': 'number'}),
        ('boolean', {'type': 'boolean'}),
        ('array', {
            'type': 'array',
            'items': {'type': 'string'},
            'minItems': random.randint(1, 5)
        }),
        ('object', {
            'type': 'object',
            'properties': {
                fake.word(): {'type': 'string'},
                fake.word(): {'type': 'integer'}
            }
        })
    ]
    return random.choice(property_types)


# 随机生成 JSON schema
def generate_random_schema(min_properties=3, max_properties=10):
    num_properties = random.randint(min_properties, max_properties)
    properties = {}
    all_prop_names = set()

    for _ in range(num_properties):
        prop_name = fake.unique.word()
        while prop_name in all_prop_names:
            prop_name = fake.unique.word()
        all_prop_names.add(prop_name)

        prop_type, prop_schema = generate_random_property()
        properties[prop_name] = prop_schema

    num_required = random.randint(0, len(properties))
    required = random.sample(list(properties.keys()), num_required)

    schema = {
        'type': 'object',
        'properties': properties,
        'required': required
    }
    return schema


# 使用 Ollama 替代 LLM 客户端类
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


# 验证生成的 JSON 是否符合 schema
def validate_json(json_str: str, schema: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        from jsonschema import validate
        data = json.loads(json_str)
        validate(instance=data, schema=schema)
        return True, ""
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, str(e)


def run_single_test(client, iteration, retries=2):
    sample_json_schema = generate_random_schema()
    prompt = f"""
    Generate a JSON object that strictly conforms to the following JSON Schema:\n\n{json.dumps(sample_json_schema, indent=2)}\n\n
    Ensure that:
    1. All required fields are present.
    2. The data types match exactly as specified (e.g., strings are quoted, numbers are not).
    3. Arrays contain the specified minimum number of items.
    ONLY provide the JSON object without any additional text or comments.
    """

    for attempt in range(retries):
        start_time = time.time()
        llm_response = client.chat(prompt)
        end_time = time.time()

        if llm_response:
            is_valid, error_message = validate_json(llm_response, sample_json_schema)
            if is_valid:
                return {
                    'iteration': iteration,
                    'schema': sample_json_schema,
                    'response': llm_response,
                    'is_valid': is_valid,
                    'error_message': error_message,
                    'time_taken': end_time - start_time
                }

        print(f"Retrying {iteration}, attempt {attempt + 1}")

    return {
        'iteration': iteration,
        'schema': sample_json_schema,
        'response': llm_response if llm_response else "",
        'is_valid': False,
        'error_message': "Failed to get valid response after retries",
        'time_taken': end_time - start_time
    }

# 主函数，执行多个测试并汇总结果
def main():
    client = Client("qwen2.5:0.5b")
    num_iterations = 100

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_single_test, client, i) for i in range(num_iterations)]

        results = []
        for future in tqdm(as_completed(futures), total=num_iterations, desc="Processing"):
            results.append(future.result())

    results.sort(key=lambda x: x['iteration'])

    total_valid = sum(1 for r in results if r['is_valid'])
    total_time = sum(r['time_taken'] for r in results)

    print(f"\nResults Summary:")
    print(f"Total tests: {num_iterations}")
    print(f"Valid responses: {total_valid}")
    print(f"Success rate: {total_valid / num_iterations * 100:.2f}%")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per test: {total_time / num_iterations:.2f} seconds")

    print("\nFailed Test Details:")
    for r in results:
        if not r['is_valid']:
            print(f"\nIteration {r['iteration']}:")
            print(f"Error: {r['error_message']}")
            print("Schema:")
            print(json.dumps(r['schema'], indent=2))
            print("Response:")
            print(r['response'])


if __name__ == "__main__":
    main()