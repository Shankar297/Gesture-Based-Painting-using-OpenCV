import numpy as np
import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()
imgCanvas = np.zeros((500,650,3),np.uint8)

while True:
    status, img = cap.read()
    img = cv2.flip(img,1)
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRBG)
    multiLandMarks = results.multi_hand_landmarks
    canDraw = False
    if multiLandMarks:
        handPoints = []
        for handsLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handsLms, mpHands.HAND_CONNECTIONS)


            for idx, lm in enumerate(handsLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                handPoints.append((cx, cy))
        if handPoints[12][1] < handPoints[10][1]:
            canDraw = False
            cv2.circle(img, handPoints[8], 15, (255,255,0))
        elif handPoints[8][1] < handPoints[6][1]:
            canDraw = True
            cv2.circle(img, handPoints[8], 15, (255,255,0), cv2.FILLED)


        if canDraw:
            cv2.circle(img, handPoints[8], 15,(255,255,0), cv2.FILLED)
            cv2.circle(imgCanvas, handPoints[8], 15,(255,255,0), cv2.FILLED)


    cv2.imshow('HandDrawing',img)
    cv2.imshow('CanvasDrawing', imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows