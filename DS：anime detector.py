import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def anime_face_detection_existing():
    cap = cv2.VideoCapture("sama-pic1.png")

    # 使用OpenCV的多个分类器组合
    cascades = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
    ]

    detectors = []
    for cascade_path in cascades:
        detector = cv2.CascadeClassifier(cascade_path)
        if not detector.empty():
            detectors.append(detector)
            print(f"加载分类器: {cascade_path.split('/')[-1]}")

    cv2.namedWindow('Anime Face Detection', cv2.WINDOW_NORMAL)

    # 定义不同的检测参数组合
    param_sets = [
        {'scale': 1.05, 'neighbors': 3, 'min_size': (40, 40)},  # 宽松参数
        {'scale': 1.1, 'neighbors': 5, 'min_size': (30, 30)},  # 中等参数
        {'scale': 1.2, 'neighbors': 7, 'min_size': (20, 20)},  # 严格参数
    ]

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            # 循环播放
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        display_frame = frame.copy()

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        all_faces = []

        # 使用不同的检测器和参数组合
        for detector in detectors:
            for params in param_sets:
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=params['scale'],
                    minNeighbors=params['neighbors'],
                    minSize=params['min_size'],
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                all_faces.extend(faces)

        # 去除重叠的检测框
        filtered_faces = []
        for (x, y, w, h) in all_faces:
            is_duplicate = False
            for (fx, fy, fw, fh) in filtered_faces:
                # 计算重叠面积
                dx = min(x + w, fx + fw) - max(x, fx)
                dy = min(y + h, fy + fh) - max(y, fy)
                if dx > 0 and dy > 0:
                    area_overlap = dx * dy
                    area1 = w * h
                    area2 = fw * fh
                    if area_overlap > 0.5 * min(area1, area2):
                        is_duplicate = True
                        break

            if not is_duplicate:
                filtered_faces.append((x, y, w, h))

        # 绘制检测结果
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        for i, (x, y, w, h) in enumerate(filtered_faces):
            color = colors[i % len(colors)]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_frame, f'Face {i + 1}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示信息
        info_text = f"Faces: {len(filtered_faces)} | Frame: {frame_count}"
        cv2.putText(display_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 调整显示大小
        display_frame = cv2.resize(display_frame, (800, 600))
        cv2.imshow('Anime Face Detection', display_frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):  # 暂停
            cv2.waitKey(0)
        elif key == ord('s'):  # 保存截图
            cv2.imwrite(f'anime_face_{frame_count}.jpg', display_frame)
            print(f"截图已保存: anime_face_{frame_count}.jpg")

    cap.release()
    cv2.destroyAllWindows()


# 运行检测
anime_face_detection_existing()