from __future__ import print_function
from imutils.object_detection import non_max_suppression
import cv2
import numpy as np
from easydict import EasyDict
from random import randint
import sys
from imutils.video import FPS

import torch
import torch.nn as nn
from torchvision import models

from utils.checkpoints import load_weights

args = EasyDict({ 

    'detector': "tracker",

    # Path Params
    'videoPath': "videos/Short4Mosaicing.mp4",

    # Player Tracking
    'classes': ["person"],
    'tracker': "CSRT",
    'trackerTypes': ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'],
    'singleTracker': False,

    # Court Line Detection
    'draw_line': False,

    # YOLOV3 Detector
    'weights': "yolov3.weights",
    'config': "yolov3.cfg",

    'COLORS': np.random.uniform(0, 255, size=(1, 3)),

    # Action Recognition
    'base_model_name': 'r2plus1d_multiclass',
    'pretrained': True,
    'lr': 0.0001,
    'start_epoch': 19,
    'num_classes': 10,
    'labels': {"0" : "block", "1" : "pass", "2" : "run", "3" : "dribble", "4" : "shoot", "5" : "ball in hand", "6" : "defense", "7" : "pick" , "8" : "no_action" , "9" : "walk" , "10" : "discard"},
    'model_path': "model_checkpoints/r2plus1d_augmented-2/",
    'history_path': "histories/history_r2plus1d_augmented-2.txt",
    'seq_length': 16,
    'vid_stride': 8,
    'output_path': "output_videos/"

})

def writeVideo(videoPath, videoFrames, playerBoxes, predictions, colors, frame_width=1280, frame_height=720, vid_stride=8):
    
    out = cv2.VideoWriter(videoPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))
    for i, frame in enumerate(videoFrames):

        # Draw Boxes
        for player in range(len(playerBoxes[0])):
            box = playerBoxes[i][player]
            # draw tracked objects

            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, colors[player], 2, 1)

            # Write Prediction
            if i // vid_stride < len(predictions[player]):
                # print(i // vid_stride)
                # print(str(predictions[player][i // vid_stride]))
                cv2.putText(frame, args.labels[str(predictions[player][i // vid_stride])], (p1[0] - 10, p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[player], 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Write the frame into the file 'output.avi'
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

def cropVideo(clip, crop_window, player=0):
    
    video = []
    #print(len(clip))
    #print(crop_window.shape)
    for i, frame in enumerate(clip):
        print(crop_window[i])
        x = int(crop_window[i][player][0])
        y = int(crop_window[i][player][1])
        w = int(crop_window[i][player][2])
        h = int(crop_window[i][player][3])

        cropped_frame = frame[y:y+h, x:x+w]
        # resize to 128x176
        try:
            resized_frame = cv2.resize(
                cropped_frame,
                dsize=(int(128),
                       int(176)),
                interpolation=cv2.INTER_NEAREST
            )
        except:
            # Use previous frame
            if len(video) == 0:
                resized_frame = np.zeros((int(176), int(128), 3), dtype=np.uint8)
            else:
                resized_frame = video[i-1]
        assert resized_frame.shape == (176, 128, 3)
        video.append(resized_frame)

    return video

def cropWindows(vidFrames, playerBoxes, seq_length=16, vid_stride=8):
    
    player_count = len(playerBoxes[0])
    player_frames = {}
    for player in range(player_count):
        player_frames[player] = []

    # How many clips in the whole video
    n_clips = len(vidFrames) // vid_stride
    # print(playerBoxes.shape)

    continue_clip = 0
    for clip_n in range(n_clips):
        crop_window = playerBoxes[clip_n*vid_stride: clip_n*vid_stride + seq_length]
        for player in range(player_count):
            if clip_n*vid_stride + seq_length < len(vidFrames):
                clip = vidFrames[clip_n*vid_stride: clip_n*vid_stride + seq_length]
                #print(" length of clip ", len(clip))
                #print(np.asarray(cropVideo(clip, crop_window, player)).shape)
                player_frames[player].append(np.asarray(cropVideo(clip, crop_window, player)))
            else:
                continue_clip = clip_n
                break
        if continue_clip != 0:
            break

    # Append to list after padding
    for i in range(continue_clip, n_clips):
        for player in range(player_count):
            crop_window = playerBoxes[vid_stride*i:]
            frames_remaining = len(vidFrames) - vid_stride * i
            clip = vidFrames[vid_stride*i:]
            player_frames[player].append(np.asarray(cropVideo(clip, crop_window, player) + [
            np.zeros((int(176), int(128), 3), dtype=np.uint8) for x in range(seq_length-frames_remaining)
        ]))

    # Check if number of clips is expected
    assert(len(player_frames[0]) == n_clips)

    return player_frames

def inference_batch(batch):
    # (batch, t, h, w, c) --> (batch, c, t, h, w)
    batch = batch.permute(0, 4, 1, 2, 3)
    return batch

def ActioRecognition(videoFrames, playerBoxes):
    frames = cropWindows(videoFrames, playerBoxes, seq_length=args.seq_length, vid_stride=args.vid_stride)
    print("Number of players tracked: {}".format(len(frames)))
    print("Number of windows: {}".format(len(frames[0])))
    print("# Frames per Clip: {}".format(len(frames[0][0])))
    print("Frame Shape: {}".format(frames[0][0][0].shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize R(2+1)D Model
    model = models.video.r2plus1d_18(pretrained=args.pretrained, progress=True)
    # input of the next hidden layer
    num_ftrs = model.fc.in_features
    # New Model is trained with 128x176 images
    # Calculation:
    model.fc = nn.Linear(num_ftrs, args.num_classes, bias=True)

    model = load_weights(model, args)

    if torch.cuda.is_available():
        # Put model into device after updating parameters
        model = model.to(device)

    model.eval()

    predictions = {}
    for player in range(len(playerBoxes[0])):
        input_frames = inference_batch(torch.FloatTensor(frames[player]))
        print('player ', player, ' input_frames ', input_frames.shape)

        input_frames = input_frames.to(device=device)
 
        with torch.no_grad():
            outputs = model(input_frames)
            _, preds = torch.max(outputs, 1)

        # print(preds.cpu().numpy().tolist())
        predictions[player] = preds.cpu().numpy().tolist()

    print('predictions ', predictions)
    
    return predictions
         