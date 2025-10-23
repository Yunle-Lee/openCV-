import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

# 使用MP4视频文件替换摄像头
video_path = "jh.mp4"  # 请替换为你的MP4文件路径
cap = cv2.VideoCapture(video_path)

# 创建人脸检测器，设置参数控制显示的点和线
detector = FaceMeshDetector(
    maxFaces=1,
    staticMode=False,  # 设置为False可以减少计算量
     # 设置为False可以简化 landmarks
    minDetectionCon=0.5,  # 检测置信度阈值
    minTrackCon=0.5  # 跟踪置信度阈值
)

while True:
    success, img = cap.read()
    if not success:
        break  # 视频播放结束

    # 检测人脸，draw=False表示不自动绘制
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]  # 只处理第一张脸

        # 只绘制关键点，不绘制所有连线
        # 可以选择性地绘制一些关键点
        key_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # 轮廓
                      33, 133,  # 左眼
                      362, 263,  # 右眼
                      61, 291,  # 嘴唇
                      13, 14,  # 鼻子
                      17, 84]  # 眉毛

        # 只绘制选定的关键点
        for point_id in key_points:
            if point_id < len(face):
                x, y = face[point_id]
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # 可选：绘制一些关键连线（简化版）
        # 例如只绘制脸部轮廓
        contour_points = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # 绘制简化的轮廓
        for i in range(len(contour_points) - 1):
            if contour_points[i] < len(face) and contour_points[i + 1] < len(face):
                start_point = tuple(face[contour_points[i]])
                end_point = tuple(face[contour_points[i + 1]])
                cv2.line(img, start_point, end_point, (255, 0, 0), 1)

    cv2.imshow('img', img)

    # 按'q'退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()