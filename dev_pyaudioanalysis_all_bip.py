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


def extract_allbip(bip_ref_path="ref_bip_isolated.wav"):
    # features of the ref
    # extract short-term features using a 50msec non-overlapping windows
    fs, s_ref = aIO.read_audio_file(bip_ref_path)
    duration = len(s_ref) / float(fs)
    win, step = 0.05, 0.05
    win_mid, step_mid = duration, 0.5
    mt_ref, st_ref, mt_n_ref = aFm.mid_feature_extraction(s_ref, fs, win_mid * fs, step_mid * fs,
                                                          win * fs, step * fs)
    
    fs = 44100
    nb_video = 16
    mt_tot = []
    

    
    
    
    for i in range(nb_video):
        video_path = 'videos - Copie/video' + str(i) + '.mp4'
        my_clip1 = mp.VideoFileClip(video_path)
        s_long = my_clip1.audio.to_soundarray(fps=fs)
        s_long = s_long[:, 0]
        win, step = 0.05, 0.05
        win_mid, step_mid = 0.4, 0.05
        mt_long, st_long, mt_n_long = aFm.mid_feature_extraction(s_long, fs, win_mid * fs, step_mid * fs,
                                                             win * fs, step * fs)
        
        
        distances = np.linalg.norm(mt_long - mt_ref, axis=0)
        arg_min_dist = np.argmin(distances)
        # plt.plot(mt_long.T[arg_min_dist])
        
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
    ####################################### Normalisation ##################################
    
    
    # for i in range(len(mt_long)):
    #     mt_long[i] = mt_long[i]/np.linalg.norm(mt_long[i])
    # for i in range (len(mt)):
    #     mt[i] = mt[i]/np.linalg.norm(mt[i])
    
    
        
    # for i in range(len(mt)):
    #     min_i = min(mt[i])
    #     for j in range(len(mt[i])):
    #         mt[i][j] = - mt[i][j] / min_i
    # for i in range(len(mt_long)):
    #     min_i = min(mt_long[i])
    #     for j in range(len(mt_long[i])):
    #         mt_long[i][j] = - mt_long[i][j] / min_i
    
    
    
    for i in range(len(mt)):
        for j in range(len(mt[i])):
            mt[i][j] = - mt[i][j] / min(mt[i])
    for i in range(len(mt_long)):
        for j in range(len(mt_long[i])):
            mt_long[i][j] = - mt_long[i][j] / min(mt_long[i])    
    

    
    
    ####################################################################################

    temps_possible = []
    
    for i in range(len(mt)):
        # plt.plot(mt[i])
        arg_min_dist = 0
        min_dist = 1000
        for j in range(len(mt_long)):
            if np.linalg.norm(mt[i]-mt_long[j]) < min_dist :
                arg_min_dist = j
                min_dist = np.linalg.norm(mt[i]-mt_long[j])
        temps_possible.append(arg_min_dist * duration_long / mt_long.shape[0])
        # print(arg_min_dist * duration_long / mt_long.shape[0])
        # print(min_dist)
    
    
    # mt_long = mt_long.T
    # time_start = arg_min_dist * duration_long / mt_long.shape[1]
    
    
    # temps_max = temps_possible[0]
    # max_apparition = 0
    # for i in range(len(temps_possible)):
    #     temps = temps_possible[i]
    #     apparition = 0
    #     for j in range(len(temps_possible)):
    #         if temps == temps_possible[j]:
    #             apparition += 1
    #     if apparition > max_apparition :
    #         max_apparition = apparition
    #         temps_max = temps
    
    median_time = np.median(temps_possible)
    temps_possible_non_aberrant = []
    aberration = 0.5
    for i in range(len(temps_possible)):
        if median_time - aberration <= temps_possible[i]:
            if temps_possible[i] <= median_time + aberration :
                temps_possible_non_aberrant.append(temps_possible[i])
    
    
    return temps_possible_non_aberrant

# video = 'VideoYoutube/natation-le-relais-francais-en-argent-derriere-les-etats-unis.mp4'




if __name__ == "__main__":
    mt = extract_allbip()
    video = 'VideoYoutube/la-finale-du-relais-4x100m-4-nages.mp4'
    start_time = extract_time_allbip(mt, video)
    print(start_time)