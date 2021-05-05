# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:13:24 2021

@author: nicol
"""

import moviepy.editor as mpy

from dev_pyaudioanalysis_start_detection import extract_time_start

vcodec = "libx264"

videoquality = "24"

compression = "slow"







def time_cut(start_time):
    m = int(start_time//60)
    s = start_time - 60*m
    if int(s)<10 :
        cut = '00:0' + str(m) + ':0' + str(s)
    else:
        cut = '00:0' + str(m) + ':' + str(s)
    return cut
    


def edit_video(loadtitle, savetitle, cut):
    #load file
    video = mpy.VideoFileClip(loadtitle)
    
    clips = []
    
    clip = video.subclip(cut[0], cut[1])
    clips.append(clip)
    
    final_clip = mpy.concatenate_videoclips(clips)
    
    
    final_clip.write_videofile(savetitle, threads=4, fps=24,
                               codec = vcodec,
                               preset = compression,
                               ffmpeg_params=["-crf",videoquality])
    video.close()
    
    

def extract_audio(savetitle, saveaudio):
    audio = mpy.AudioFileClip(savetitle)
    audio.write_audiofile(saveaudio, 44100)
    
    
if __name__ == "__main__":
    for i in range(15):
        loadtitle = "videos2/video" + str(i) + '.mp4'
        start_time = extract_time_start(loadtitle)
        bip_duration = 0.3401360544217687
        cut = (time_cut(start_time), time_cut(start_time + bip_duration))
        savetitle = "bips/clip" + str(i) + '.mp4'
        saveaudio = "bips/BIP" + str(i) + '.wav'
        edit_video(loadtitle, savetitle, cut)
        extract_audio(savetitle, saveaudio)
    