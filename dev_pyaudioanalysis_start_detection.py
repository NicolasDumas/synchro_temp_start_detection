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


def extract_time_start(video_path, bip_ref_path="ref_bip_isolated.wav"):
    # features of the ref
    # extract short-term features using a 50msec non-overlapping windows
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5
    mt_ref, st_ref, mt_n_ref = aFm.mid_feature_extraction(s_ref, fs, win_mid * fs, step_mid * fs,
                                                          win * fs, step * fs)
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

    # compute the distance and get the minimum
    
 ######################################## Normalisation ################################   
    mt_long = mt_long.T
    for i in range(len(mt_long)):
        mt_long[i] = mt_long[i]/np.linalg.norm(mt_long[i])
    mt_long = mt_long.T
    mt_ref = mt_ref/np.linalg.norm(mt_ref)
    
    # mt_long = mt_long.T
    # for i in range(len(mt_long)):
    #     mt_long[i] = mt_long[i] * min(mt_ref) / min(mt_long[i])
    # mt_long = mt_long.T
    
    # for i in range(len(mt_ref)):
    #     mt_ref[i] = - mt_ref[i] / min(mt_ref)
    # mt_long = mt_long.T
    # for i in range(len(mt_long)):
    #     for j in range(len(mt_long[i])):
    #         mt_long[i][j] = - mt_long[i][j] / min(mt_long[i])
    # mt_long = mt_long.T
    
    # mt_long = mt_long.T
    # for i in range(len(mt_long)):
    #     for j in range(len(mt_long[i])):
    #         mt_long[i][j] = mt_long[i][j] / mt_ref[j]
    # mt_long = mt_long.T
    # for i in range(len(mt_ref)):
    #     mt_ref[i] = mt_ref[i] / mt_ref[i]
    
##########################################################################################

    distances = np.linalg.norm(mt_long - mt_ref, axis=0)
    plt.plot(distances)
    
    arg_min_dist = np.argmin(distances)
    time_start = np.argmin(distances) * duration_long / mt_long.shape[1]
    
    # plt.plot(mt_ref)
    # plt.plot(mt_long.T[arg_min_dist])
    return time_start


def extract_possible_time_start(video_path, seuil, bip_ref_path="Rebond_TT.wav"):
    # features of the ref
    # extract short-term features using a 50msec non-overlapping windows
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5
    mt_ref, st_ref, mt_n_ref = aFm.mid_feature_extraction(s_ref, fs, win_mid * fs, step_mid * fs,
                                                          win * fs, step * fs)
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

    # compute the distance and get the minimum
    






    distances = np.linalg.norm(mt_long - mt_ref, axis=0)
    plt.plot(distances)
    
    time_possible = []
    for i in range(len(distances)):
        if distances[i] < seuil :
            if distances[i] <= distances[i-1] and distances[i] <= distances[i+1]:
                time_possible.append(i * duration_long / mt_long.shape[1])
                # plt.plot(mt_long.T[i])  
    return time_possible





if __name__ == "__main__":
    video = 'videosEspadon/50 PAP MESSIEURS FINALE MIGNON PIRON PONTI MANAUDOU KORSTANJE GROUSSET VERMOREL PUTS.mp4'
    start_time = extract_time_start(video)
    print(start_time)
    # seuil = 4
    # possible_time = extract_possible_time_start(video, seuil)
    # print(possible_time)
    






