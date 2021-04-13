# Tools for swimming video analysis

This repository contains different scripts to help handle and analyse videos from swimming races. Those scripts are used to do jobs on a server. 

## Prod

*prepro_startDetection_auto.py*: Input a video, extracts its sound and detect the moment where the sound is the closest to *ref_bip_isolated.wav*. 
This allows to detect the start of a race. 

*stitch_2_videos.py*: Inputs two fixed videos and computes the bird's eye view of the swimming pool.

*print_positions_video.py*: Inputs a bird's eye view video and the tracking of the swimmers (oddata format) and display the positions on the video.

## Dev 

*zoom_positions.py*: Inputs the tracking of the swimmers and two videos to zoom on the swimmer on the video.

*two_lines_against_each_other.py*: Inputs two bird's eye view videos and outputs a video of the chosen lanes competing against each other.

#### The rest of the scripts are used for testing or research purposes
