from doctest import OutputChecker
import librosa as lb
import soundfile as sf
import sys
from os.path import exists
import random

def convert_to_pcm_16k(file):
    y, s = lb.load(file, sr=16000)
    sf.write(file, y, s, subtype='PCM_16')

    # for i in range(10000):
    #     index = random.randrange(18934446)
    #     input_file = f'data/en/clips/common_voice_en_{index}.mp3'
    #     if exists(input_file):
    #         output_file = f'data/en_16k/common_voice_en_{index}.wav'
    #         y, s = lb.load(input_file, sr=16000)
    #         sf.write(output_file, y, s, subtype='PCM_16')

    print('Done!')

if __name__ == '__main__':
    convert_to_pcm_16k(sys.argv[1:])