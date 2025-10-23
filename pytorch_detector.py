import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


def preprocess_for_anime(frame):
    """使用PyTorch进行图像预处理"""
    # 转换为PyTorch tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    tensor_img = transform(frame).unsqueeze(0)

    # 简单的图像增强
    # 增加对比度
    tensor_img = torch.clamp(tensor_img * 1.2, 0, 1)

    # 转回numpy
    enhanced = tensor_img.squeeze(0).permute(1, 2, 0).numpy()
    enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced


def enhanced_detection_with_torch():
    cap = cv2.VideoCapture("two-sama.mp4")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cv2.namedWindow('Enhanced Anime Detection', cv2.WINDOW_NORMAL)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 方法1: 原始图像检测
        gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_original = face_cascade.detectMultiScale(
            gray_original, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )

        # 方法2: 增强后图像检测
        enhanced_frame = preprocess_for_anime(frame)
        gray_enhanced = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2GRAY)

        # 调整坐标（因为图像被resize了）
        scale_x = frame.shape[1] / 256
        scale_y = frame.shape[0] / 256

        faces_enhanced = face_cascade.detectMultiScale(
            gray_enhanced, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
        )

        # 转换坐标回原图尺寸
        faces_enhanced_scaled = []
        for (x, y, w, h) in faces_enhanced:
            x_orig = int(x * scale_x)
            y_orig = int(y * scale_y)
            w_orig = int(w * scale_x)
            h_orig = int(h * scale_y)
            faces_enhanced_scaled.append((x_orig, y_orig, w_orig, h_orig))

        # 合并检测结果
        all_faces = list(faces_original) + faces_enhanced_scaled

        # 绘制结果
        result_frame = frame.copy()
        for i, (x, y, w, h) in enumerate(all_faces):
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_frame, f'Detected {i + 1}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示信息
        cv2.putText(result_frame, f'Faces: {len(all_faces)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        result_frame = cv2.resize(result_frame, (800, 600))
        cv2.imshow('Enhanced Anime Detection', result_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 运行增强检测
enhanced_detection_with_torch()