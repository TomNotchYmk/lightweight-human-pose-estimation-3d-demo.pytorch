from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import time

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
from pygame import mixer
from enum import Enum
from playsound import playsound

class Body(Enum):
    neck = 0
    nose = 1
    center = 2
    l_shoulder = 3
    l_elbow = 4
    l_wrist = 5
    l_hip = 6
    l_knee = 7
    l_ankle = 8
    r_shoulder = 9
    r_elbow = 10
    r_wrist = 11
    r_hip = 12
    r_knee = 13
    r_ankle = 14
    r_eye = 15
    l_eye = 16
    r_ear = 17
    l_ear = 18

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

def move_to_root(pose_3d):
    root_point = np.multiply(np.add(pose_3d[Body.l_ankle.value], pose_3d[Body.r_ankle.value]), np.array([0.5, 0.5, 0.5]))
    for i in range(len(pose_3d)):
        pose_3d[i] = np.subtract(pose_3d[i], root_point)

def print_keypoints(pose_3d):
    print("3D points inferred:\n")
    buffer = pose_3d.copy()
    buffer = buffer.reshape((19, buffer.size//19))
    for i in range(len(buffer)):
        print(Body(i).name, " ", buffer[i])
    print("\n")

def get_thigh_angle(pose_3d):
    mid_hip = np.multiply(np.add(pose_3d[Body.l_hip.value], pose_3d[Body.r_hip.value]), np.array([0.5, 0.5, 0.5]))
    mid_knee = np.multiply(np.add(pose_3d[Body.l_knee.value], pose_3d[Body.r_knee.value]), np.array([0.5, 0.5, 0.5]))
    hip_relative = np.subtract(mid_hip, mid_knee)
    hip_xy = hip_relative.copy()
    hip_xy[-1] = 0
    return np.arccos(np.linalg.norm(hip_xy)/np.linalg.norm(hip_relative)) * 180 / np.pi

if __name__ == '__main__':
    mixer.init()
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-t', '--time', default=-1, type=int)
    # parser.add_argument('-l', '--limit-time',
    #                     help='whether to do the squats in a minite or not',
    #                     type=str, required=True)
    args = parser.parse_args()

    stride = 8
    from modules.inference_engine_pytorch import InferenceEnginePyTorch
    net = InferenceEnginePyTorch("human-pose-estimation-3d.pth", 'GPU', use_tensorrt=False)

    canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    frame_provider = VideoReader(0)
    is_video = True
    base_height = 256
    fx = -1

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0

    # thigh count data section
    thigh_up = True
    thigh_count = 0
    thigh_time_stamp = time.time()

    if args.time > 0:
        playsound(os.path.join("audio", "go.mp3"))
        start_time = time.time()

    for frame in frame_provider:
        current_time = cv2.getTickCount()
        if frame is None:
            break
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        
        
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            move_to_root(poses_3d[-1])
            print_keypoints(poses_3d[-1])
            thigh_angle = get_thigh_angle(poses_3d[-1])

            if thigh_angle > 75:
                if thigh_up == False:
                    thigh_time_stamp = time.time()
                thigh_up = True
            elif thigh_angle < 60:
                if thigh_up == True and time.time() - thigh_time_stamp > 0.5:
                    thigh_count += 1
                    mixer.music.load(os.path.join("audio", (str(thigh_count)+".mp3")))
                    mixer.music.play()
                thigh_up = False

            if args.time > 0 and time.time() - start_time > args.time:
                playsound(os.path.join("audio", "total_count.mp3"))
                break

            # print("thigh_count == ", thigh_count)
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
        plotter.plot(canvas_3d, poses_3d, edges)
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imshow('ICV 3D Human Pose Estimation', frame)

        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1
        if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                plotter.plot(canvas_3d, poses_3d, edges)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1

    playsound(os.path.join("audio", "total_count.mp3"))
    playsound(os.path.join("audio",  (str(thigh_count)+".mp3")))