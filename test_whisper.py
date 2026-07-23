import whisper

model = whisper.load_model("base")

result = model.transcribe("test.mp4")

print(result["text"])
