import os
import torch
from TTS.api import TTS

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)

emotion_prompts = {
    "happy":     "said cheerfully, ",
    "sad":       "said in a sad tone, ",
    "angry":     "said angrily, ",
    "neutral":   "",
}

voice_map = {
    "male":   "voices/male.wav",
    "female": "voices/female.wav",
}

def generate_speech(text, gender, emotion, output_path="output.wav"):
    speaker_wav = voice_map.get(gender, "voices/female.wav")

    if not os.path.exists(speaker_wav):
        raise FileNotFoundError(f"Voice file not found: {speaker_wav}")

    prefix = emotion_prompts.get(emotion, "")
    styled_text = prefix + text

    tts.tts_to_file(
        text=styled_text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=output_path
    )

    print(f"✅ Speech saved to {output_path}")