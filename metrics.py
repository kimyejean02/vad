import numpy as np
from sklearn.metrics import precision_recall_fscore_support as metrics
from re import S
import vad_engines.silero as silero
import vad_engines.cobra as cobra
import vad_engines.webrtc as webrtc
import soundfile as sf
import wave, sys
from synthesize import generate_soundscapes
import os
import librosa as lb

#IGNORE WARNING ABOUT SOX

'''
FIRST, WE MUST CREATE SOME AUDIO DATA TO TEST OUR MODELS ON. WE USE SCAPER TO CREATE A MIXTURE OF BACKGROUND TRACKS
AND FOREGROUND AUDIO SAMPLES. 


SCAPER WILL RANDOMLY SELECT AUDIO FROM THE AUDIO/BACKGROUND FOLDER AND VOICE SAMPLES FROM AUDIO/FOREGROUND.
THE RESULTING SOUNDSCAPES ARE WRITTEN TO AUDIO/SOUNDSCAPES. (THIS DIRECTORY MUST EXIST BEFORE SCAPER RUNS)

ALTER THE BELOW PARAMETERS AS NECESSARY
'''

outfolder = 'audio/soundscapes/'
fg_folder = 'audio/foreground/'
bg_folder = 'audio/background/'
n_soundscapes = 5
ref_db = -50
duration = 10.0
min_events = 1
max_events = 2
event_time_dist = 'truncnorm'
event_time_mean = 5.0
event_time_std = 2.0
event_time_min = 0.0
event_time_max = 10.0
source_time_dist = 'const'
source_time = 0
event_duration_dist = 'uniform'
event_duration_min = 1.5
event_duration_max = 4.0
snr_dist = 'uniform'
snr_min = 2
snr_max = 10
pitch_dist = 'uniform'
pitch_min = -1.0
pitch_max = 1.0
time_stretch_dist = 'uniform'
time_stretch_min = 0.8
time_stretch_max = 1.2

#this call will overwrite whatever exists in audio/soundscapes
generate_soundscapes(outfolder=outfolder,
                    fg_folder=fg_folder,
                    bg_folder=bg_folder,
                    n_soundscapes=n_soundscapes,
                    ref_db=ref_db,
                    duration=duration,
                    min_events=min_events,
                    max_events=max_events,
                    event_time_dist=event_time_dist,
                    event_time_mean=event_time_mean,
                    event_time_std=event_time_std,
                    event_time_min=event_time_min,
                    event_time_max =event_time_max,
                    source_time_dist=source_time_dist,
                    source_time=source_time,
                    event_duration_dist=event_duration_dist,
                    event_duration_min=event_duration_min,
                    event_duration_max=event_duration_max,
                    snr_dist=snr_dist,
                    snr_min=snr_min,
                    snr_max=snr_max,
                    pitch_dist=pitch_dist,
                    pitch_min=pitch_min,
                    pitch_max=pitch_max,
                    time_stretch_dist=time_stretch_dist,
                    time_stretch_min=time_stretch_min,
                    time_stretch_max=time_stretch_max)

def convert_to_pcm_16k(file):
    y, s = lb.load(file, sr=16000)
    sf.write(file, y, s, subtype='PCM_16')

with wave.open('./audio/soundscapes/soundscape_unimodal0.wav', 'rb') as wf:    
        nchannels, sampwidth, framerate, nframes, comptype, compname = wf.getparams()

ground_truth_sum = np.zeros((n_soundscapes, nframes))
cobra_preds_sum = np.zeros((n_soundscapes, nframes))
silero_preds_sum = np.zeros((n_soundscapes, nframes))
webrtc_preds_sum = np.zeros((n_soundscapes, nframes))

def generate_gt_timestamps(gt_filepath, num_frames, SAMPLING_RATE):
    gt_timestamps = np.zeros((num_frames))    
    for line in open(gt_filepath, 'r'):
        split = line.split("/t")
        if len(split) == 3:
            start, end, label = line.split("\t")
            start = (int)(float(start) * SAMPLING_RATE)
            end = (int)(float(end) * SAMPLING_RATE)
            gt_timestamps[start:end] = 1
    return gt_timestamps

#ACTUALLY DO THE VAD PREDICTION AND GROUND TRUTH CALCULATIONS HERE


# directory name
dirname = './audio/soundscapes' 
# scanning the directory to get required files
i = 0
for files in os.scandir(dirname):
    if files.path.endswith('.wav'):
        #convert the audio track to 16,000 for compliance with all VAD engines
        SAMPLING_RATE = 16000
        convert_to_pcm_16k(files.path)
        gt_filepath = files.path[:-3] +'txt'
        gt_timestamps = generate_gt_timestamps(gt_filepath, nframes, SAMPLING_RATE)
        
        filepath = files.path
        silero_voice_timestamps = silero.get_voice_timestamps(filepath, SAMPLING_RATE)
        cobra_threshold = 0.2
        cobra_voice_timestamps = cobra.get_voice_timestamps(filepath, threshold=cobra_threshold)
        webrtc_voice_timestamps = webrtc.get_voice_timestamps(filepath)
        
        return_time = False
        if return_time:
            silero_voice_timestamps = frame_to_time(silero_voice_timestamps, SAMPLING_RATE)
            cobra_voice_timestamps = frame_to_time(cobra_voice_timestamps, SAMPLING_RATE)
            webrtc_voice_timestamps = frame_to_time(webrtc_voice_timestamps, SAMPLING_RATE)
        
        cobra_preds = np.zeros(nframes)
        silero_preds = np.zeros(nframes)
        webrtc_preds = np.zeros(nframes)

        for interval in cobra_voice_timestamps:
            start = interval['start']
            end = interval['end']
            cobra_preds[start:end] = 1
    
        for interval in silero_voice_timestamps:
            start = interval['start']
            end = interval['end']
            silero_preds[start:end] = 1

        for interval in webrtc_voice_timestamps:
            start = interval['start']
            end = interval['end']
            webrtc_preds[start:end] = 1
        
        ground_truth_sum[i] = gt_timestamps
        cobra_preds_sum[i] = cobra_preds
        silero_preds_sum[i] = silero_preds
        webrtc_preds_sum[i] = webrtc_preds
        
        i+=1

cobra_precision, cobra_recall, cobra_f, cobra_support = \
            metrics(ground_truth_sum.T, cobra_preds_sum.T, beta=2, zero_division=1)
silero_precision, silero_recall, silero_f, silero_support = \
            metrics(ground_truth_sum.T, silero_preds_sum.T, beta=2, zero_division=1)
webrtc_precision, webrtc_recall, webrtc_f, webrtc_support = \
            metrics(ground_truth_sum.T, webrtc_preds_sum.T, beta=2, zero_division=1)

cobra_precision, cobra_recall, cobra_f, cobra_support = \
            np.mean(cobra_precision), np.mean(cobra_recall), np.mean(cobra_f), np.mean(cobra_support)

silero_precision, silero_recall, silero_f, silero_support = \
            np.mean(silero_precision), np.mean(silero_recall), np.mean(silero_f), np.mean(silero_support)

webrtc_precision, webrtc_recall, webrtc_f, webrtc_support = \
            np.mean(webrtc_precision), np.mean(webrtc_recall), np.mean(webrtc_f), np.mean(webrtc_support)

print('STATS: ')
print('- precision')
print('- recall')
print('- f-score')
print()

print('COBRA')
print(cobra_precision)
print(cobra_recall)
print(cobra_f)
print()

print('SILERO')
print(silero_precision)
print(silero_recall)
print(silero_f)
print()


print('WEBRTC')
print(webrtc_precision)
print(webrtc_recall)
print(webrtc_f)