import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load WAV file
speech, rate = sf.read("sample.wav")

# Process input
inputs = processor(speech, sampling_rate=rate, return_tensors="pt").input_values

# Inference
with torch.no_grad():
    logits = model(inputs).logits

# Decode prediction
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])

print("Transcription:", transcription)
