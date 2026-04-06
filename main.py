from tts import generate_speech

text    = "I lost my job today"
emotion = "sad"
gender  = "male"

try:
    generate_speech(text, gender, emotion)
except Exception as e:
    print(f"❌ Error: {e}")