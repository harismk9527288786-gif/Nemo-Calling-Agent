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

def main():
    with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    client = genai.Client(api_key=api_key)

    system_instruction = """
You are an energetic, warm Indian phone calling assistant.
Speak in natural Devanagari Hinglish. 1-2 short sentences max.
"""

    prompt = "हेलो, मुझे पर्सनल लोन के बारे में जानना है।"
    print(f"=== Testing Gemini 3.7 Flash Streaming (Zero Thinking Budget for Low Latency) ===")
    print(f"User: {prompt}\n")

    for attempt in range(3):
        try:
            t0 = time.perf_counter()
            first_token_time = None
            full_text = ""

            # Disable thinking for ultra-fast streaming voice responses
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )

            stream = client.models.generate_content_stream(
                model="gemini-3.7-flash",
                contents=prompt,
                config=config
            )

            print("Agent: ", end="", flush=True)
            for chunk in stream:
                if first_token_time is None and chunk.text:
                    first_token_time = time.perf_counter()
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_text += chunk.text

            t_end = time.perf_counter()
            ttft = (first_token_time - t0) * 1000 if first_token_time else 0
            total_time = (t_end - t0) * 1000
            print(f"\n\n⚡ Time to First Token (TTFT): {ttft:.1f} ms")
            print(f"⚡ Total Generation Time: {total_time:.1f} ms")
            break
        except errors.ServerError as e:
            print(f"\n[Attempt {attempt+1}] 503 Spike: {e.message}. Retrying in 2s...")
            time.sleep(2)
        except Exception as e:
            print(f"\n[Attempt {attempt+1}] Error: {e}")
            break

if __name__ == "__main__":
    main()
