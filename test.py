import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier(r"C:\Users\shilp\OneDrive\Documents\Luminar\Internship\BSL\Model\keras_model.h5",
                         r"C:\Users\shilp\OneDrive\Documents\Luminar\Internship\BSL\Model\labels.txt")
offset = 20
imgSize = 451
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W",
          "X", "Y", "Z"]

folder = 'Data/Z'
while True:
    success, img = cap.read()
    imgOutput = img.copy()
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

        # Crop and resize for both hands
        imgCrop = img[y_min-offset:y_max+offset, x_min-offset:x_max+offset]
        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
        imgWhite[:imgSize, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite)
        print(prediction, index)

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        else:
            print("Invalid crop dimensions")

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
