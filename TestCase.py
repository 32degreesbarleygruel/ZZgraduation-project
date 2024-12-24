import ollama
res=ollama.chat(
    model="phi3.5:3.8b",
    stream=False,
    messages=[
        {
        "role": "user","content": "你是谁，用中文"
        }
    ],
    options={
        "temperature":0
    }
)
print(res)