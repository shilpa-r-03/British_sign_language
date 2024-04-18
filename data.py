import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 600
counter = 0

folder = 'Data/Z'
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands and len(hands) == 2:  # Ensure there are exactly 2 hands
        hand1, hand2 = hands[0], hands[1]

        x1, y1, w1, h1 = hand1['bbox']
        x2, y2, w2, h2 = hand2['bbox']

        # Calculate a bounding box that includes both hands
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y_min-offset:y_max+offset, x_min-offset:x_max+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = imgCropShape[1] / imgCropShape[0]

        if aspectRatio > 1:
            k = imgSize / imgCropShape[1]
            wCal = int(k * imgCropShape[1])  # Ensure wCal is an integer
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = int((imgSize - wCal) / 2)  # Ensure wGap is an integer
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / imgCropShape[0]
            hCal = int(k * imgCropShape[0])  # Ensure hCal is an integer
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = int((imgSize - hCal) / 2)  # Ensure hGap is an integer
            imgWhite[hGap:hCal+hGap, :] = imgResize

        if imgCropShape[0] > 0 and imgCropShape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        else:
            print("Invalid crop dimensions")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

    if key == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
