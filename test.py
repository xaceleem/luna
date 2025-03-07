import soundfile as sf

wav_path = "/Users/evankaiden/Documents/SimpleTransfromerTTS-copy/content/speech_dataset/LJSpeech-1.1/wavs/LJ028-0440.wav"


import torchaudio

waveform, sample_rate = torchaudio.load(wav_path)
