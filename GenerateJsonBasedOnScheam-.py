import json

from TheOutputFormatIsJson import Emotion

model = "phi3.5:3.8b"
extra_body = {
    "guided_json": Emotion.model_json_schema(),
    "guided_whitespace_pattern": ""
}

sample_json_schema = {
    "type": "object",
    "properties": {
        "must": {
            "type": "string"
        },
        "pattern": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 2
        },
        "most": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 3
        },
        "glass": {
            "type": "string"
        },
        "hospital": {
            "type": "object",
            "properties": {
                "guy": {
                    "type": "string"
                },
                "hospital": {
                    "type": "integer"
                }
            }
        },
        "act": {
            "type": "number"
        },
        "note": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 1
        },
        "hundred": {
            "type": "string"
        },
        "place": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "minItems": 3
        }
    },
    "required": [
        "pattern",
        "must",
        "act"
    ]
}

prompt = f"Generate a JSON object that conforms to the following JSONSchema:\n\n{json.dumps(sample_json_schema, indent=2)}\n\nEnsure that the generated JSON includes all required fields and adheres to any specified constraints."

# 生成的 JSON 对象
generated_json = {
    "must": "do this",
    "pattern": ["one", "two"],
    "most": ["eat", "sleep", "code"],
    "glass": "clear",
    "hospital": {"guy": "John Doe", "hospital": 123},
    "act": 42,
    "note": ["remember to breathe"],
    "hundred": "100",
    "place": ["home", "work", "gym"]
}

# 打印生成的 JSON 对象
print(json.dumps(generated_json, indent=2))