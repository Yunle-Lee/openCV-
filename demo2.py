import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture("two-sama.mp4")
detector = FaceMeshDetector(maxFaces=2)

# 设置窗口为可调整
cv2.namedWindow('img', cv2.WINDOW_NORMAL)

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img)

    # 直接设置固定大小（宽度，高度）
    img_resized = cv2.resize(img, (8000, 6500))  # 你可以调整这里的数字

    cv2.imshow('img', img_resized)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()