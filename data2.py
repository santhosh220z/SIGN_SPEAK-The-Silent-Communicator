import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Allow detection of up to two hands
offset = 20
counter = 0

# Update the folder path to save images based on the gesture
folder = "C:/Users/santh/Desktop/hand_sign/hand_gesture_pred/data/"

while True:
    success, img = capture.read()
    if not success:
        print("Failed to capture image. Retrying...")
        continue  # Skip the rest of the loop if the image capture failed

    hands, img = detector.findHands(img)
    cv2.imshow("Live Feed", img)  # Ensure live feed is always displayed
    
    key = cv2.waitKey(1)
    if key == ord('c') and hands and len(hands) >= 1:  # Save only if two hands are detected
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', img)
        print(f"Image {counter} saved")
    elif key == ord('q'):  # Check for 'q' key to quit
        break  # Exit the loop

# Release the capture after exiting the loop
capture.release()
cv2.destroyAllWindows()
