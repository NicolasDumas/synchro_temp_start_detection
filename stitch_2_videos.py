import numpy as np
import json
import cv2
import argparse


def get_index(list_dict, vid_name):
    """helper to read the json file."""
    index = -1
    for i in range(len(list_dict)):
        if list_dict[i]['name'] == vid_name:
            index = i
    return index


def stitch(save_path, video1, video2, time_difference, src_pts1, dest_pts1, src_pts2, dest_pts2, start_side_vid):
    """Stitches video1 and video2 based on the source and destination points
    Saves the result at save_path"""
    # hard coded size of the reference image
    size_image_ref = (900, 360)

    # size of the final image
    shape_output_img = (1920, 1080)

    # we need to convert the points of the calibration to make them correspond the destination image size
    dest_pts1[:, 0] = dest_pts1[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_pts1[:, 1] = dest_pts1[:, 1] * shape_output_img[1] / size_image_ref[1]
    dest_pts2[:, 0] = dest_pts2[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_pts2[:, 1] = dest_pts2[:, 1] * shape_output_img[1] / size_image_ref[1]

    # generating the homography matrices
    hm1 = cv2.getPerspectiveTransform(src_pts1, dest_pts1)
    hm2 = cv2.getPerspectiveTransform(src_pts2, dest_pts2)

    # video reading
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    time_shift = round(time_difference*fps)

    # output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, (shape_output_img[0], shape_output_img[1]))

    # Check if camera opened successfully
    if cap1.isOpened() == False:
        print("Error opening video 1 stream or file")
    if cap2.isOpened() == False:
        print("Error opening video 2 stream or file")

    # we read frames until synchronised
    for _ in range(abs(time_shift)):
        if time_shift > 0: cap1.read()
        else: cap2.read()

    while (cap1.isOpened()) and (cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 is not True or ret2 is not True:
            break
        else:
            # transformation of the frame
            left_trans = cv2.warpPerspective(frame1, hm1, (shape_output_img[0], shape_output_img[1]))
            right_trans = cv2.warpPerspective(frame2, hm2, (shape_output_img[0], shape_output_img[1]))

            # stitch them together
            out_top = np.where(right_trans != 0, right_trans, left_trans)
            out.write(out_top)
    cap1.release()
    cap2.release()
    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for stitching two videos together.')
    parser.add_argument('--json', help='Path of the json of the race')
    parser.add_argument('--videog', help='Path of the left video')
    parser.add_argument('--videod', help='Path of the right video')
    parser.add_argument('--out', help='Path to output the result with name of the file in .mp4')
    args = parser.parse_args()

    save_path = args.out
    video1 = args.videog
    video1_name = video1.split('/')[-1]
    video2 = args.videod
    video2_name = video2.split('/')[-1]
    json_path = args.json

    # extraction of the calibration points
    with open(json_path) as json_file:
        json_course = json.load(json_file)

    index_vid1 = get_index(json_course['videos'], video1_name)
    index_vid2 = get_index(json_course['videos'], video2_name)
    src_pts1 = np.float32(json_course['videos'][index_vid1]['srcPts'])
    dest_pts1 = np.float32(json_course['videos'][index_vid1]["destPts"])
    src_pts2 = np.float32(json_course['videos'][index_vid2]['srcPts'])
    dest_pts2 = np.float32(json_course['videos'][index_vid2]["destPts"])

    # time difference should be obtained from the json
    time_left = json_course['videos'][index_vid1]['start_moment']
    time_right = json_course['videos'][index_vid2]['start_moment']
    time_difference = time_left - time_right

    # side where the swimmers start on the video
    start_side = json_course['videos'][index_vid1]['start_side']

    # run the function
    fps = stitch(save_path, video1, video2, time_difference, src_pts1, dest_pts1, src_pts2, dest_pts2, start_side)

    # change and save the json
    from_above_info = {'name': save_path.split('/')[-1],
                       'type_video': 'vueDessus',
                       'start_moment': min(time_right, time_left),
                       'one_is_up': json_course['videos'][index_vid1]['one_is_up'],
                       'generated_from': [video1_name, video2_name],
                       'fps': fps,
                       'start_side': start_side}
    # if the info on the video exists we delete the information:
    index_from_above = get_index(json_course['videos'], save_path.split('/')[-1])
    if index_from_above > -1:
        json_course['videos'][index_from_above] = from_above_info
    else:
        json_course['videos'].append(from_above_info)

    with open(args.json, 'w') as outfile:
        json.dump(json_course, outfile, indent=4)

    # # to run :
    # python3 pipeline-tracking-docs/synchro_prepro/stitch_2_videos.py
    # --json test_videos/2021_Nice_brasse_50_finaleA_dames.json
    # --videog test_videos/2021_Nice_brasse_50_finaleA_dames_fixeGauche.mp4
    # --videod test_videos/2021_Nice_brasse_50_finaleA_dames_fixeDroite.mp4
    # --out test_videos/2021_Nice_brasse_50_finaleA_dames_from_above.mp4
