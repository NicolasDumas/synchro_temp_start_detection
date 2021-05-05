from pyAudioAnalysis import MidTermFeatures as aFm
from pyAudioAnalysis import audioBasicIO as aIO
import moviepy.editor as mp
import numpy as np
# import argparse
# import json
import matplotlib.pyplot as plt


# def get_index(list_dict, vid_name):
#     """helper to read the json file."""
#     for i in range(len(list_dict)):
#         if list_dict['videos'][i]['name'] == vid_name:
#             return i


def extract_time_start(bip_ref_path="ref_bip_isolated.wav"):
    # features of the ref
    # extract short-term features using a 50msec non-overlapping windows
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5
    mt_ref, st_ref, mt_n_ref = aFm.mid_feature_extraction(s_ref, fs, win_mid * fs, step_mid * fs,
                                                          win * fs, step * fs)
    
    fs = 44100
    nb_video = 15
    mt_tot = []
    
    for i in range(nb_video):
        video_path = 'videos2/video' + str(i) + '.mp4'
        my_clip1 = mp.VideoFileClip(video_path)
        s_long = my_clip1.audio.to_soundarray(fps=fs)
        s_long = s_long[:, 0]
        win, step = 0.05, 0.05
        win_mid, step_mid = 0.4, 0.05
        mt_long, st_long, mt_n_long = aFm.mid_feature_extraction(s_long, fs, win_mid * fs, step_mid * fs,
                                                             win * fs, step * fs)
        distances = np.linalg.norm(mt_long - mt_ref, axis=0)
        arg_min_dist = np.argmin(distances)
        plt.plot(mt_long.T[arg_min_dist])
        
        mt_tot.append(mt_long.T[arg_min_dist])

    
    return mt_tot


def extract_time_allbip(mt, video_path, bip_ref_path="ref_bip_isolated.wav"):
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5

    # extraction on the long signal
    my_clip1 = mp.VideoFileClip(video_path)
    fs = 44100
    s_long = my_clip1.audio.to_soundarray(fps=fs)
    s_long = s_long[:, 0]
    duration_long = len(s_long) / float(fs)

    # extract short-term features using a 50msec non-overlapping windows
    win, step = 0.05, 0.05
    win_mid, step_mid = 0.4, 0.05
    mt_long, st_long, mt_n_long = aFm.mid_feature_extraction(s_long, fs, win_mid * fs, step_mid * fs,
                                                             win * fs, step * fs)



    mt_long = mt_long.T
    # compute the distance and get the minimum
    
    
    for i in range(len(mt)):
        arg_min_dist = 0
        min_dist = 1000
        for j in range(len(mt_long)):
            if np.linalg.norm(mt[i]-mt_long[j]) < min_dist :
                arg_min_dist = j
                min_dist = np.linalg.norm(mt[i]-mt_long[j])
        print(arg_min_dist * duration_long / mt_long.shape[0])
        print(min_dist)
    
    
    # mt_long = mt_long.T
    # time_start = arg_min_dist * duration_long / mt_long.shape[1]
    
    return 0









# if __name__ == "__main__":
#     mt = extract_time_start()
#     video = 'videos/50_DOS_M_FA_lowered.mp4'
#     extract_time_allbip(mt, video)
