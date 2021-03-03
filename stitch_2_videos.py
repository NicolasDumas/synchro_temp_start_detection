import numpy as np
import json
import cv2
import argparse


def get_index(list_dict, vid_name):
    """helper to read the json file."""
    for i in range(len(list_dict)):
        if list_dict[i]['name'] == vid_name:
            return i


def stitch(save_path, video1, video2, time_difference, src_pts1, dest_pts1, src_pts2, dest_pts2):
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
    file_name = save_path.split('/')[-2] + '_from_above.mp4'
    out = cv2.VideoWriter(save_path + file_name, fourcc, fps, (shape_output_img[0], shape_output_img[1]))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser for stitching two videos together.')
    parser.add_argument('--json', help='Path of the json of the race')
    parser.add_argument('--videog', help='Path of the left video')
    parser.add_argument('--videod', help='Path of the right video')
    parser.add_argument('--out', help='Path to output the result')
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

    # run the function
    stitch(save_path, video1, video2, time_difference, src_pts1, dest_pts1, src_pts2, dest_pts2)
