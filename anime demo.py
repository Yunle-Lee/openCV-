import cv2
from anime_face_detector import create_detector


def anime_face_detection():
    cap = cv2.VideoCapture("two-sama.mp4")
    detector = create_detector('yolov3')  # 使用YOLO模型

    cv2.namedWindow('Anime Face Detection', cv2.WINDOW_NORMAL)

    while True:
        success, img = cap.read()
        if not success:
            break

        # 检测动漫面部
        preds = detector(img)

        # 绘制结果
        for pred in preds:
            bbox = pred['bbox']  # [x, y, w, h]
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'Anime Face', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img_resized = cv2.resize(img, (800, 600))
        cv2.imshow('Anime Face Detection', img_resized)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


anime_face_detection()