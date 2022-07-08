from re import S
import vad_engines.silero as silero
import vad_engines.cobra as cobra
import vad_engines.webrtc as webrtc
import soundfile as sf
import wave, sys


#rewrite audio to 16 bit
# data, samplerate = sf.read(FILE_PATH)
# sf.write(FILE_PATH, data, samplerate, subtype='PCM_16')
def frame_to_time(timestamps, sampling_rate):
    res = []
    for timestamp in timestamps:
        new_stamp = {}
        new_stamp['start'] = timestamp['start'] / sampling_rate
        new_stamp['end'] = timestamp['end'] / sampling_rate
        res.append(new_stamp)
    return res

def process_labels(file):
    #process labels generated from scaper
    #TODO:combine overlapping timestamps
    return

if __name__ == '__main__':
    filepath = sys.argv[1]
    if len(sys.argv) > 2:
        return_time = sys.argv[2]
    else:
        return_time = True # default to returning time instead of frames
    SAMPLING_RATE = 16000   

    silero_voice_timestamps = silero.get_voice_timestamps(filepath, SAMPLING_RATE)
    cobra_threshold = 0.2
    cobra_voice_timestamps = cobra.get_voice_timestamps(filepath, threshold=cobra_threshold)
    webrtc_voice_timestamps = webrtc.get_voice_timestamps(filepath)

    if return_time:
        silero_voice_timestamps = frame_to_time(silero_voice_timestamps, SAMPLING_RATE)
        cobra_voice_timestamps = frame_to_time(cobra_voice_timestamps, SAMPLING_RATE)
        webrtc_voice_timestamps = frame_to_time(webrtc_voice_timestamps, SAMPLING_RATE)


    print(f'Silero:\n{silero_voice_timestamps}')
    print(f'Cobra:\n{cobra_voice_timestamps}')
    print(f'WebRTC:\n{webrtc_voice_timestamps}')


