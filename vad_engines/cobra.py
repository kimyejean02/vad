from pickle import FRAME
import pvcobra as cobra
import wave, struct

def read_file(file_name, sample_rate):
    wav_file = wave.open(file_name, mode="rb")
    channels = wav_file.getnchannels()
    num_frames = wav_file.getnframes()

    if wav_file.getframerate() != sample_rate:
        raise ValueError("Audio file should have a sample rate of %d. got %d" % (sample_rate, wav_file.getframerate()))

    samples = wav_file.readframes(num_frames)
    wav_file.close()

    frames = struct.unpack('h' * num_frames * channels, samples)

    if channels == 2:
        print("Picovoice processes single-channel audio but stereo file is provided. Processing left channel only.")

    return frames[::channels]

def get_voice_timestamps(file_path, threshold=.5):
    #create an instance of the engine
    handle = cobra.create(access_key="09N18zdSX5XT8HD2fI7uwhYPyM/QizmP+QIRjZO/QFeBKFrBjzj8EA==")
    audio = read_file(file_path, handle.sample_rate)
    num_frames = len(audio) // handle.frame_length
    print(f'num_frames: {num_frames}')
    timestamps = []
    latest_end = 0
    latest_start = 0
    for i in range(num_frames):
        frame = audio[i*handle.frame_length:(i+1)*handle.frame_length]
        speech_prob = handle.process(frame)
        if speech_prob > threshold:
            timestamp = float(i * handle.frame_length) #/ float(handle.sample_rate) to get in seconds
            if abs(timestamp - latest_end) > 5000: #start a new segment if the timestamps are more than 5000 frames apart
                timestamps.append({'start': latest_start, 'end': latest_end})
                latest_start = int(timestamp)
            latest_end = int(timestamp)
    return timestamps
