import speech_recognition as sr

def get_voice_timestamps(file_path, sampling_rate, merge_speech=False):

    r = sr.Recognizer()
    clip = sr.AudioFile(file_path)
    with clip as source:
        r.adjust_for_ambient_noise(source)
        audio = r.record(source, duration=20)
        print(r.recognize_google(audio))