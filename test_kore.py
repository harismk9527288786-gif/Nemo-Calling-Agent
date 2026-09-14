import sys
import asyncio
import wave
import pygame
from google import genai
from google.genai import types

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

async def main():
    with open("gemini_api_key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()

    client = genai.Client(api_key=api_key)
    voice_name = "Kore"
    prompt = "अरे वाह सर! आप बिल्कुल सही जगह आए हैं, मैं आपको अभी हमारे बेस्ट डिस्काउंट और स्पेशल ऑफर्स बताती हूँ!"

    print(f"=== Generating & Playing Voice: {voice_name} ===")
    print(f"Prompt: {prompt}\n")

    config = types.LiveConnectConfig(
        response_modalities=['AUDIO'],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
        system_instruction="You are an energetic, friendly, and enthusiastic Indian calling assistant. Speak in conversational Hinglish with natural warmth and sweetness."
    )

    audio_chunks = []
    print("Connecting to Gemini Live...")
    async with client.aio.live.connect(model="gemini-2.5-flash-native-audio-latest", config=config) as session:
        await session.send(input=prompt, end_of_turn=True)
        print("Receiving native audio stream...")
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
        out_file = "gemini_voice_kore.wav"
        with wave.open(out_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(full_pcm)

        print(f"Saved: {out_file} ({len(full_pcm)} bytes)")
        print("Playing audio via speakers...")
        pygame.mixer.init(frequency=24000)
        pygame.mixer.music.load(out_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        print("Done playing Kore voice!")

if __name__ == "__main__":
    asyncio.run(main())
