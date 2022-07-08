
import torch
from pprint import pprint

def get_voice_timestamps(file_path, sampling_rate, merge_speech=False):

    #potential for increasing efficiency / use GPUS and batching
    torch.set_num_threads(1)


    #initialization
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    wav = read_audio(file_path, sampling_rate=sampling_rate)


    # get speech timestamps from full audio file
    #TIME STAMP / SAMPLING RATE = seconds
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    for timestamp in speech_timestamps:
        start_time = timestamp['start'] / sampling_rate
        end_time = timestamp['end'] / sampling_rate

    return speech_timestamps