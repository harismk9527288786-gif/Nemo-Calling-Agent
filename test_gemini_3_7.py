import sys
import time
from google import genai
from google.genai import types
from google.genai import errors

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_model(client, model_name, test_prompts, system_instruction):
    print(f"\n========================================================")
    print(f"  Testing Model: {model_name}")
    print(f"========================================================")
    
    for idx, prompt in enumerate(test_prompts, 1):
        print(f"\n[Turn {idx}] User: {prompt}")
        success = False
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                    )
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                print(f"Agent ({model_name}): {response.text.strip()}")
                print(f"⚡ Latency: {latency_ms:.1f} ms")
                success = True
                break
            except errors.ServerError as e:
                print(f"  [Attempt {attempt+1}] 503 Busy: {e.message}. Retrying in 2s...")
                time.sleep(2)
            except Exception as e:
                print(f"  [Attempt {attempt+1}] Error: {e}")
                time.sleep(1)
        if not success:
            print(f"  ❌ Model {model_name} failed turn {idx}")

def main():
    with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    client = genai.Client(api_key=api_key)

    system_instruction = """
You are an energetic, warm, and highly motivated Indian phone calling assistant.
Persona & Rules:
1. Tone: Enthusiastic, friendly, empathetic, and eager to help.
2. Language: Speak in natural, everyday conversational Hinglish (Devanagari script with common English words).
3. Expressiveness: Use genuine human warmth and conversational interjections ("अरे बिल्कुल सर!", "हाँ जी बिल्कुल!").
4. Length: SHORT AND CRISP (1-2 sentences max).
"""

    test_prompts = [
        "हेलो, मुझे एक होम लोन के बारे में जानकारी चाहिए थी। क्या आप मदद कर सकते हैं?",
        "मुझे अपना क्रेडिट कार्ड ब्लॉक करवाना है तुरंत, चोरी हो गया है!",
        "Can you explain your interest rates in simple terms?"
    ]

    models_to_test = ["gemini-3.7-flash", "gemini-2.5-flash"]
    for m in models_to_test:
        test_model(client, m, test_prompts, system_instruction)

if __name__ == "__main__":
    main()
