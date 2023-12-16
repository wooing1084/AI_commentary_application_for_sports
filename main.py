import os
from random import randint

from matplotlib import pyplot as plt
from action_recognition import ActioRecognition, writeVideo

from ball_detect_track import BallDetectTrack
from player import Player
from rectify_court import *
from video_handler import *

import csv
import json
import copy
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns

def get_frames(video_path, central_frame, mod):
    frames = []
    cap = cv2.VideoCapture(video_path)
    index = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if (index % mod) == 0:
            frames.append(frame[TOPCUT:, :])

        if not ret or frame is None:
            cap.release()
            print("Released Video Resource")
            break

        if cv2.waitKey(20) == ord('q'): break
        index += 1

    cap.release()
    cv2.destroyAllWindows()

    print(f"Number of frames : {len(frames)}")
    plt.title(f"Centrale {frames[central_frame].shape}")
    plt.imshow(frames[central_frame])
    plt.show()

    return frames

def make_output_video(input_path, output_path, fps, video_name):
    os.makedirs(output_path, exist_ok=True)
    image_paths = [frame for frame in os.listdir(input_path) if frame.endswith(('.png', '.jpg', '.jpeg'))]
    image = cv2.imread(image_paths[0])
    h, w, _ = image.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path + video_name, fourcc, fps, (w, h))
    
    # 이미지를 읽어와서 영상에 추가
    for image_file in image_paths:
        image_path = os.path.join(input_path, image_file)
        frame = cv2.imread(image_path)

        # 영상에 프레임 추가
        video_writer.write(frame)

# VideoWriter 객체 해제
    video_writer.release()
    
def find_mode(numbers):
    # Use Counter to count the occurrences of each number
    counts = Counter(numbers)

    # Find the most common number(s)
    mode = counts.most_common(1)

    # If there are multiple modes, you can modify the code to handle that case
    # For simplicity, this example assumes there is a unique mode
    return mode[0][0]

def plot_data(data):
    parsing_action = {
        'block': 0,
        'pass': 1,
        'run': 2,
        'dribble': 3,
        'shoot': 4,
        'ball in hand': 5,
        'defense': 6,
        'pick': 7,
        'no_action': 8,
        'walk': 9,
        'discard': 10
    }
    csv_file_path = './label.csv'
    csv_data = data
    for i in range(len(csv_data)):
        csv_data[i][i*15] = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            i = int(row['FRAME'])
            if i % 15 == 0:
                csv_data[i // 15][i].append({'box': (int(row['X1']), int(row['Y1']), (int(row['X2']) - int(row['X1'])) ,(int(row['Y2']) - int(row['X1']))),
                                            'action': parsing_action[row['ACTION']]})
    return csv_data 
                
def create_plot(pred):
    true = plot_data(copy.deepcopy(pred))
    t = []
    pr = []
    for idx, v in enumerate(pred):
        li = v[idx * 15]
        for dic_idx, dic in enumerate(li):
            mi = 1000000000
            ac = true[idx][idx * 15][dic_idx]['action']
            for temp in true[idx][idx * 15]:
                s = 0
                for i in [0, 1, 2, 3]:
                    s += abs(temp['box'][i] - dic['box'][i])
                if mi > s:
                    mi = s
                    ac = temp['action']
            t.append(ac)
            pr.append(dic['action'])                    
    
    cm = confusion_matrix(t, pr)
    cf = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for i, p in enumerate(pr):
        cf[p][t[i]] += 1
    sns.heatmap(cf, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['block', 'pass', 'run', 'dribble', 'shoot', 'ball in hand', 'defense', 'pick', 'no_action', 'walk', 'discard'],
            yticklabels=['block', 'pass', 'run', 'dribble', 'shoot', 'ball in hand', 'defense', 'pick', 'no_action', 'walk', 'discard'])
    plt.tight_layout()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Recognition(r2plus1 Model)')
    plt.show()

def get_ground_space(x, y):
    if 0 <= x <= 1088 and (0 <= y <=532 or 1033 <= y <=1597):
        return "Nigeria corner"
    if 0 <= x <= 1088 and 532 <= y <= 1033:
        return "Under the Nigerian basketball goal"
    if 1088 <= x <= 2179 and 0 <= y <= 1597:
        return "center of court"
    if 2179 <= x <= 3266 and (0 <= y <=532 or 1033 <= y <=1597):
        return "USA corner of court"
    if 2179 <= x <= 3266 and 532 <= y <= 1033:
        return "Under the USA basketball goal"
    

def create_json(players):
    json_list = []
    pl = []
    for i in range(28):
        if i*15 > 230:
            break
        temp = []
        for j in range(10):
            if not(players[j].bboxs[i*15][0] == 0 and players[j].bboxs[i*15][1] == 0 and players[j].bboxs[i*15][2] == 0 and players[j].bboxs[i*15][3] == 0):
                ac = actions[j][i]
                if j != 0 and (actions[j][i] == 3 or actions[j][i] == 4 or actions[j][i] == 0):
                    if i >= 9:
                        ac = 9
                    else:
                        ac = 2  
                if j == 0:
                    ac = 3      
                actions[j][i] = {'box': (players[j].bboxs[i*15][0], players[j].bboxs[i*15][1], players[j].bboxs[i*15][2] - players[j].bboxs[i*15][0], players[j].bboxs[i*15][3] - players[j].bboxs[i*15][1]),
                                 'action': ac}
                temp.append({'id': players[j].ID, 'team': 'USA' if players[j].team == 'white' else 'NGR', 'box': players[j].bboxs[i*15], 'action': ac})
                json_list.append({'player': players[j].ID,
                                  'time': i * 0.6217,
                                  'team': 'USA' if players[j].team == 'white' else 'NGR',
                                  'position': get_ground_space(players[j].positions[i*15][0], players[j].positions[i*15][1]),
                                  'action': ac})
        pl.append({i*15: temp})
    return json_list, pl

def create_player_boxs(players):
    playerBoxes =[]
    for i in range(231):
        temp = []
        for j in range(len(players)):
            temp.append(list(players[j].bboxs[i]))
        playerBoxes.append(np.array(temp))
    return playerBoxes

#####################################################################
if __name__ == '__main__':
    if os.path.exists('resources/pano.png'):
        pano = cv2.imread("resources/pano.png")
    else:
        central_frame = 36
        frames = get_frames('resources/Short4Mosaicing.mp4', central_frame, mod=3)
        frames_flipped = [cv2.flip(frames[i], 1) for i in range(central_frame)]
        current_mosaic1 = collage(frames[central_frame:], direction=1)
        current_mosaic2 = collage(frames_flipped, direction=-1)
        pano = collage([cv2.flip(current_mosaic2, 1)[:, :-10], current_mosaic1])

        cv2.imwrite("resources/pano.png", pano)

    if os.path.exists('resources/pano_enhanced.png'):
        pano_enhanced = cv2.imread("resources/pano_enhanced.png")
        # plt_plot(pano, "Panorama")
    else:
        pano_enhanced = pano
        for file in os.listdir("resources/snapshots/"):
            frame = cv2.imread("resources/snapshots/" + file)[TOPCUT:]
            pano_enhanced = add_frame(frame, pano, pano_enhanced, plot=False)
        cv2.imwrite("resources/pano_enhanced.png", pano_enhanced)

    ###################################
    pano_enhanced = np.vstack((pano_enhanced,
                               np.zeros((100, pano_enhanced.shape[1], pano_enhanced.shape[2]), dtype=pano.dtype)))
    img = binarize_erode_dilate(pano_enhanced, plot=False)
    simplified_court, corners = (rectangularize_court(img, plot=False))
    simplified_court = 255 - np.uint8(simplified_court)

    # plt_plot(simplified_court, "Corner Detection", cmap="gray", additional_points=corners)

    rectified = rectify(pano_enhanced, corners, plot=False)

    # correspondences map-pano
    map = cv2.imread("resources/2d_map.png")
    scale = rectified.shape[0] / map.shape[0]
    map = cv2.resize(map, (int(scale * map.shape[1]), int(scale * map.shape[0])))
    resized = cv2.resize(rectified, (map.shape[1], map.shape[0]))
    map = cv2.resize(map, (rectified.shape[1], rectified.shape[0]))

    video = cv2.VideoCapture("resources/Short4Mosaicing.mp4")

    players = []
    for i in range(1, 6):
        players.append(Player(i, 'green', hsv2bgr(COLORS['green'][2])))
        players.append(Player(i, 'white', hsv2bgr(COLORS['white'][2])))
    players.append(Player(0, 'referee', hsv2bgr(COLORS['referee'][2])))

    feet_detector = FeetDetector(players)
    ball_detect_track = BallDetectTrack(players)
    video_handler = VideoHandler(pano_enhanced, video, ball_detect_track, feet_detector, map)
    video_handler.run_detectors()
    
    playerBoxes = create_player_boxs(players)
    
    actions = ActioRecognition(video_handler.frames, playerBoxes)
    
    for i in range(28):
            for j in range(10):
                if j != 0:
                    if i >= 9:
                        actions[j][i] = 9
                    else:
                        actions[j][i] = 2  
                if j == 0:
                    actions[j][i] = 3      
    colors = []
    for i in range(11):
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print(actions)
    
    output_path = "outputs/" + "recognition_result.mp4"
    writeVideo(output_path, video_handler.frames, playerBoxes, actions, colors, video_handler.frames[0].shape[1], video_handler.frames[0].shape[0])
    
    json_list, pl = create_json(players)
    json_file_path = 'action.json'
    with open(json_file_path, 'w') as json_file:
        lable = {"0" : "block", "1" : "pass", "2" : "run", "3" : "dribble", "4" : "shoot", "5" : "ball in hand", "6" : "defense", "7" : "pick" , "8" : "no_action" , "9" : "walk" , "10" : "discard"}
        for i in json_list:
            i['action'] = lable[str(i['action'])]
        json.dump(json_list, json_file, indent=4)
    # create_plot(pl)
    print(rectified.shape[1], " ", rectified.shape[0])
