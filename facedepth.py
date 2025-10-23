import cv2
import cvzone
from charset_normalizer import detect
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture("neuro sama.mp4")
detector = FaceMeshDetector(maxFaces=10)

while True:
    success ,img= cap.read()
    img,faces = detector.findFaceMesh(img)
    cv2.imshow('img',img)
    cv2.waitKey(8)
