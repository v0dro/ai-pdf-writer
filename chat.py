import requests

OLLAMA_URL = "http://0.0.0.0:11434/api/chat"
MODEL = "mistral"

# Initialize chat
chat_history = [
    {"role": "system", "content": "You are a chatbot that collects user info: name, email, and preferred appointment time. Be brief and ask one question at a time."},
]

def ask_ollama(prompt):
    chat_history.append({"role": "user", "content": prompt})
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "messages": chat_history})
    reply = response.json()["message"]["content"]
    print("Bot:", reply)
    chat_history.append({"role": "assistant", "content": reply})
    return reply

# Chat loop to gather info
print("Chatbot started. Type 'done' to finish.")
user_data = {}

while True:
    user_input = input("You: ")
    if user_input.lower() == 'done':
        break
    bot_reply = ask_ollama(user_input)

# Ask the model to return structured data
summary = ask_ollama("Please summarize the collected data in JSON format.")

import json
try:
    user_data = json.loads(summary)
except Exception as e:
    print("Failed to parse JSON:", e)
    exit()

print("\nCollected Data:", user_data)
