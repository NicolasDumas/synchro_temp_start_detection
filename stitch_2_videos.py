import numpy as np
import json
import cv2


def get_index(list_dict, vid_name):
    for i in range(len(list_dict)):
        if list_dict[i]['name'] == vid_name:
            return i


def stitch(save_path, video1, video2, time_difference, src_pts1, dest_pts1, src_pts2, dest_pts2):
    size_image_ref = (900, 360)
    shape_output_img = (1024, 1024)
    dest_pts1[:, 0] = dest_pts1[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_pts1[:, 1] = dest_pts1[:, 1] * shape_output_img[1] / size_image_ref[1]
    dest_pts2[:, 0] = dest_pts2[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_pts2[:, 1] = dest_pts2[:, 1] * shape_output_img[1] / size_image_ref[1]
    hm1 = cv2.getPerspectiveTransform(src_pts1, dest_pts1)
    hm2 = cv2.getPerspectiveTransform(src_pts2, dest_pts2)
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    time_shift = round(time_difference*fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file_name = save_path.split('/')[-2] + '_from_above.mp4'
    out = cv2.VideoWriter(save_path + file_name, fourcc, fps, (int(1024), int(1024)))

    # Check if camera opened successfully
    if cap1.isOpened() == False:
        print("Error opening video 1 stream or file")
    if cap2.isOpened() == False:
        print("Error opening video 2 stream or file")

    for _ in range(abs(time_shift)):
        if time_shift > 0: cap1.read()
        else: cap2.read()

    while (cap1.isOpened()) and (cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 is not True or ret2 is not True:
            break
        else:
            left_trans = cv2.warpPerspective(frame1, hm1, (shape_output_img[0], shape_output_img[1]))
            right_trans = cv2.warpPerspective(frame2, hm2, (shape_output_img[0], shape_output_img[1]))
            out_top = np.where(right_trans != 0, right_trans, left_trans)
            out.write(out_top)
    cap1.release()
    cap2.release()

if __name__ == "__main__":
    save_path = "videos/"
    video1 = "2021_Nice_brasse_50_serie3_dames_fixeGauche.mp4"
    video2 = "2021_Nice_brasse_50_serie3_dames_fixeDroite.mp4"
    json_path = save_path + "2021_Nice_brasse_50_serie3_dames.json"
    with open(json_path) as json_file:
        json_course = json.load(json_file)
    src_pts1 = np.float32(json_course['videos'][get_index(json_course['videos'], video1)]['srcPts'])
    dest_pts1 = np.float32(json_course['videos'][get_index(json_course['videos'], video1)]["destPts"])
    src_pts2 = np.float32(json_course['videos'][get_index(json_course['videos'], video2)]['srcPts'])
    dest_pts2 = np.float32(json_course['videos'][get_index(json_course['videos'], video2)]["destPts"])

    time_difference = 1.34
    stitch(save_path, save_path+video1, save_path+video2, time_difference, src_pts1, dest_pts1, src_pts2, dest_pts2)
