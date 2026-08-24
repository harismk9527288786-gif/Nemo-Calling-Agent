import sys
import asyncio
import wave
import pygame
from google import genai
from google.genai import types

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = genai.Client(api_key=api_key)

voices_to_test = [
    ("Puck", "अरे हाय सर! मैं एसके फर्नीचर से बात कर रहा हूँ। बताइए मैं आपकी क्या हेल्प करूँ?"),
    ("Aoede", "नमस्ते सर! बिल्कुल, हमारे पास सभी तरह के मॉडर्न सोफा और डाइनिंग टेबल की बेहतरीन रेंज उपलब्ध है।"),
    ("Kore", "अरे वाह! आप बिल्कुल सही जगह आए हैं, मैं आपको अभी हमारे बेस्ट डिस्काउंट और ऑफर्स बताती हूँ!"),
    ("Fenrir", "हाँ जी सर! आपकी डिलीवरी बिल्कुल टाइम पर और सेफली आपके घर पहुँच जाएगी, आप बिल्कुल बेफिक्र रहिए।")
]

async def test_voices():
    pygame.mixer.init(frequency=24000)
    
    for voice_name, prompt in voices_to_test:
        print(f"\n" + "="*50)
        print(f"  Testing Voice: {voice_name}")
        print(f"  Prompt: {prompt}")
        print("="*50)
        
        config = types.LiveConnectConfig(
            response_modalities=['AUDIO'],
            enable_affective_dialog=True,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
            system_instruction="You are an energetic, warm, and highly motivated Indian calling assistant. Speak in natural conversational Hinglish. Respond with genuine emotion and motivation."
        )

        audio_chunks = []
        try:
            async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as session:
                await session.send(input=prompt, end_of_turn=True)
                
                async for response in session.receive():
                    server_content = response.server_content
                    if server_content is not None:
                        model_turn = server_content.model_turn
                        if model_turn is not None:
                            for part in model_turn.parts:
                                if part.inline_data and part.inline_data.data:
                                    audio_chunks.append(part.inline_data.data)
                        if server_content.turn_complete:
                            break

            if audio_chunks:
                full_pcm = b"".join(audio_chunks)
                out_file = f"gemini_voice_{voice_name.lower()}.wav"
                with wave.open(out_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(full_pcm)
                    
                print(f"Playing {voice_name} ({len(full_pcm)} bytes)...")
                pygame.mixer.music.load(out_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                pygame.mixer.music.unload()

        except Exception as e:
            print(f"Error on {voice_name}: {e}")

    pygame.mixer.quit()
    print("\nVoice showcase complete!")

if __name__ == "__main__":
    asyncio.run(test_voices())
