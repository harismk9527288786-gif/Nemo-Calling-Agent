from google import genai

with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
    key = f.read().strip()

client = genai.Client(api_key=key)

print("Listing available models on your Gemini project...")
for m in client.models.list():
    if "bidi" in str(m.supported_actions) or "live" in m.name.lower() or "flash" in m.name.lower():
        print(f"Model: {m.name} | Supported: {m.supported_actions}")
