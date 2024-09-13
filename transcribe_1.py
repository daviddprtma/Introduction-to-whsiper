import whisper

model = whisper.load_model("base")
transcript = model.transcribe("sample_audio/audio.wav")
print(result['text'])