import cv2
import numpy as np
import pandas as pd
import argparse
import json


def get_index(list_dict, vid_name):
    """helper to read the json file."""
    index = -1
    for i in range(len(list_dict)):
        if list_dict[i]['name'] == vid_name:
            index = i
    return index


def zoom_two_videos(videog, videod, start_timeg, start_timed, swimmer_data, hm_right, hm_left, save_path, size_box,
                    start_size_vid):
    """Input: right and left video of the race, start_time gauche, start time droite, données du nageur
    Il doit y avoir autant de donnée qu'il y a de frame dans la vidéo"""
    capg = cv2.VideoCapture(videog)
    capd = cv2.VideoCapture(videod)
    fps = capg.get(cv2.CAP_PROP_FPS)

    # output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(save_path, fourcc, fps, size_box)

    # invert the homography matrix
    new_hm_right = np.linalg.inv(hm_right)
    new_hm_left = np.linalg.inv(hm_left)

    time_shiftg = round((start_timeg - 1) * fps)
    time_shiftd = round((start_timed - 1) * fps)
    compt = 0
    # we read frames until synchronised
    for _ in range(abs(time_shiftg)):
        capg.read()
    for _ in range(abs(time_shiftd)):
        capd.read()

    while capd.isOpened() and capg.isOpened():
        retd, framed = capd.read()
        retg, frameg = capg.read()

        # mask = np.zeros((1920, 1080), dtype="uint8")
        # cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)

        if retd is not True or compt >= len(swimmer_data):
            break
        else:
            # zoom
            x = swimmer_data[compt][1]
            to_save = np.zeros((size_box[1], size_box[0], 3)).astype(np.uint8)

            # to choose the side of the video
            if x != -1 and x < 25:
                # convert x to a position that the homography maps
                if start_size_vid == 'right':
                    x = (50 - x) * 1920 / 50
                else:
                    x = x * 1920 / 50

                w = size_box[0]
                y = 1080 * 3 / 8
                h = size_box[1]

                # compute coordinates using linear algebra
                coor_maind = np.dot(new_hm_right, np.array([x, y, 1]))
                coor_maind = (coor_maind / coor_maind[-1]).astype(int)
                x_side, y_side = coor_maind[0], coor_maind[1]
                # to_save = np.zeros(size_box)
                to_save = framed[y_side - h//2:y_side + h//2, x_side - w//2:x_side + w//2]

            elif x != -1:
                # convert x to a position that the homography maps
                # coor vue dessus
                if start_size_vid == 'right':
                    x = (50 - x) * 1920 / 50
                else:
                    x = x * 1920 / 50

                w = size_box[0]
                y = 1080 * 3 / 8
                h = size_box[1]
                # using the opencv functions
                to_transform = np.float32([[[x, y]]])  # np.array([x, y, 1])
                coorg = cv2.perspectiveTransform(to_transform, new_hm_left)
                coorg = np.squeeze(coorg).astype(int)
                x_side, y_side = coorg[0], coorg[1]
                # to_save = np.zeros(size_box)
                to_save = frameg[y_side - h // 2:y_side + h // 2, x_side - w // 2:x_side + w // 2]
        # write the new image (it will be black if x = -1)
        out.write(to_save)

        compt += 1

    capd.release()
    capg.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for zoom on one swimmer.')
    parser.add_argument('--json', help='Path of the json of the race')
    parser.add_argument('--videog', help='Path of the left video')
    parser.add_argument('--videod', help='Path of the right video')
    parser.add_argument('--out', help='path to output the resulting video.')
    parser.add_argument('--lane', help='lane number to extract the zoom from.', default='3')
    parser.add_argument('--csv', help='Csv with the tracking data to use to zoom.')
    parser.add_argument('--type_data', help='auto or manuelle for ', default='auto')
    args = parser.parse_args()

    # get the info from the json
    with open(args.json) as json_file:
        json_course = json.load(json_file)
    name_of_video_to_get_info_gauche = args.json.split('/')[-1].split('.')[0] + '_fixeGauche.mp4'
    name_of_video_to_get_info_droite = args.json.split('/')[-1].split('.')[0] + '_fixeDroite.mp4'
    index_vidg = get_index(json_course['videos'], name_of_video_to_get_info_gauche)
    index_vidd = get_index(json_course['videos'], name_of_video_to_get_info_droite)
    start_side = json_course['videos'][index_vidg]['start_side']
    start_timeg = json_course['videos'][index_vidg]['start_moment']
    start_timed = json_course['videos'][index_vidd]['start_moment']
    fps = int(json_course['videos'][index_vidd]['fps'])

    # get the homography matrices : first the correspondance points
    src_ptsg = np.float32(json_course['videos'][index_vidg]['srcPts'])
    dest_ptsg = np.float32(json_course['videos'][index_vidg]["destPts"])
    src_ptsd = np.float32(json_course['videos'][index_vidd]['srcPts'])
    dest_ptsd = np.float32(json_course['videos'][index_vidd]["destPts"])
    # we need to convert the points of the calibration to make them correspond the destination image size
    shape_output_img = (1920, 1080)
    size_image_ref = (900, 360)
    dest_ptsg[:, 0] = dest_ptsg[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_ptsg[:, 1] = dest_ptsg[:, 1] * shape_output_img[1] / size_image_ref[1]
    dest_ptsd[:, 0] = dest_ptsd[:, 0] * shape_output_img[0] / size_image_ref[0]
    dest_ptsd[:, 1] = dest_ptsd[:, 1] * shape_output_img[1] / size_image_ref[1]
    # generating the homography matrices
    hm_left = cv2.getPerspectiveTransform(src_ptsg, dest_ptsg)
    hm_right = cv2.getPerspectiveTransform(src_ptsd, dest_ptsd)

    # converting the data to use them easily in the function with a numpy array and the index is the frame index
    if start_side == 'right':
        start_frame = round((start_timed - 1) * fps)
    else:
        start_frame = round((start_timeg - 1) * fps)
    if args.type_data == 'auto':
        data = pd.read_csv(args.csv)  # id, frame_number, swimmer, x1, x2, y1, y2, event, cycles
        data = data.to_numpy()
        all_swimmers = []
        for i in range(8):
            swimmer = np.squeeze(data[np.argwhere(data[:, 2] == i)])[:, (1, 3)]
            to_interpolate = pd.DataFrame(swimmer, columns=['x', 'y'])
            to_interpolate = to_interpolate.replace('-1', np.nan)
            to_interpolate.loc[start_frame] = -1.1
            # print(swimmer)
            data_to_print = to_interpolate[['x', 'y']].interpolate(method='index')
            data_to_print['x'] = data_to_print['x'].astype(int)
            data_to_print['y'] = data_to_print['y'].astype(int)
            data_to_print = data_to_print.to_numpy()
            all_swimmers.append(swimmer) # just change to swimmer to cancel the interpolation
        all_swimmers = np.array(all_swimmers)
    elif args.type_data == 'manuel':
        data = pd.read_csv(args.csv)
        # some info on the data
        name_of_video_to_get_info_above = args.json.split('/')[-1].split('.')[0] + '_from_above.mp4'
        index_vid_above = get_index(json_course['videos'], name_of_video_to_get_info_above)
        start_time_above = json_course['videos'][index_vid_above]['start_moment']
        data['frame_number'] = ((data['time'] - start_time_above + 1)*fps).astype(int)
        data.set_index('frame_number', inplace=True)
        data = data.groupby(by='id').apply(lambda x : x.reindex(range(0, max(x.index) + 1))) # .interpolate(method='index')
        data.loc[(slice(None),start_frame)] = -1.1
        data = data.groupby(level='id').apply(lambda x: x.to_numpy())
        all_swimmers = [data[i] for i in range(len(data))]
        print(all_swimmers[0])

    # let's compute the zoom
    size_box = (384, 256)
    zoom_two_videos(args.videog, args.videod, start_timeg, start_timed, all_swimmers[int(args.lane)], hm_right, hm_left, args.out,
                    size_box, start_side)

    # information of the video in the json
    # change and save the json
    zoom_info = {'name': args.json.split('/')[-2] + '_zoom_' + str(args.lane) + '.mp4',
                    'type_video': 'zoom',
                    'start_moment': 1,
                    'one_is_up': json_course['videos'][index_vidg]['one_is_up'],
                    'generated_from': [name_of_video_to_get_info_gauche, name_of_video_to_get_info_droite, args.csv.split('/')[-1]],
                    'fps': json_course['videos'][index_vidg]['fps'],
                    'start_side': start_side
                    }
    index_zoom = get_index(json_course['videos'], args.json.split('/')[-2] + '_zoom_' + str(args.lane) + '.mp4')
    if index_zoom > -1:
        json_course['videos'][index_zoom] = zoom_info
    else:
        json_course['videos'].append(zoom_info)

    with open(args.json, 'w') as outfile:
        json.dump(json_course, outfile, indent=4)

    ## Todo : gestion des différences de fps entre la vidéo qui a servit à l'analyse et la vidéo sur laquelle on fait le zoom
        # faire une augmentation des index puis une interpolation ?
        # faire juste un convertisseur
    ## Todo : faire en sorte que ça marche avec les données de l'annotation manuelle
    ## Todo : supprimer les autres lines
    ## todo : faire tourner openCV sur une vidéo pour comparer le qualité

    # command to launch the code
    # python3 zoom_on_swimmers.py
    # --json /home/amigo/Bureau/data/kazan2015_david/50_Brasse_Women_Final/2015_Kazan_brasse_dames_50_finale.json
    # --videog /home/amigo/Bureau/data/kazan2015_david/50_Brasse_Women_Final/50_B_W_F_lowered.mp4
    # --videod /home/amigo/Bureau/data/kazan2015_david/50_Brasse_Women_Final/50_B_W_F_Cd_lowered.mp4
    # --out /home/amigo/Bureau/data/kazan2015_david/50_Brasse_Women_Final/2015_Kazan_brasse_dames_50_finale_zoom.mp4
    # --csv /home/amigo/Bureau/data/kazan2015_david/50_Brasse_Women_Final/2015_Kazan_brasse_dames_50_finale_automatique.csv