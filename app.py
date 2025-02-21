import cv2
import numpy as np
import streamlit as st
import time
import pyttsx3
import threading
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Initialize text-to-speech engine
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("SIGN SPEAK")

# Load Hand Detector and Classifier
detector = HandDetector(maxHands=2, detectionCon=0.8)
classifier = Classifier(
    'C:/Users/santh/Desktop/hand_sign/hand_gesture_pred/model/converted_keras (1)2/keras_model.h5',
    'C:/Users/santh/Desktop/hand_sign/hand_gesture_pred/model/converted_keras (1)2/labels.txt'
)

# Parameters
offset = 15
imgsize = 224
labels = ['good job', 'hello', 'house', 'lose', 'Love', 'no', 'play', 'thank you', 'victory', 'yes']

# Streamlit Session State Initialization
if "run_webcam" not in st.session_state:
    st.session_state["run_webcam"] = False
if "detected_words" not in st.session_state:
    st.session_state["detected_words"] = ""
if "last_spoken_word" not in st.session_state:
    st.session_state["last_spoken_word"] = ""

# Streamlit Controls
def start_webcam():
    st.session_state["run_webcam"] = True

def stop_webcam():
    st.session_state["run_webcam"] = False

FRAME_WINDOW = st.empty()

# Place buttons at the top
col1, col2 = st.columns(2)
with col1:
    st.button("Start Webcam", on_click=start_webcam)
with col2:
    st.button("Stop Webcam", on_click=stop_webcam)

# Sentence Display (Now Below Buttons)
sentence_display = st.empty()

# Webcam streaming loop
if st.session_state["run_webcam"]:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    prev_time = 0

    while st.session_state["run_webcam"]:
        success, image = cap.read()
        if not success:
            st.warning("Failed to access webcam.")
            break

        imgOutput = image.copy()
        hands, img = detector.findHands(image, draw=True)
        detected_texts = []  # Store detected hand signs

        if len(hands) == 2:  # Two-handed detection
            hand1, hand2 = hands
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            # Create a combined bounding box
            x_min = min(x1, x2) - offset
            y_min = min(y1, y2) - offset
            x_max = max(x1 + w1, x2 + w2) + offset
            y_max = max(y1 + h1, y2 + h2) + offset

            # Crop the region containing both hands
            imgCrop = image[y_min:y_max, x_min:x_max]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
                aspectRatio = (y_max - y_min) / (x_max - x_min)

                if aspectRatio > 1:
                    k = imgsize / (y_max - y_min)
                    wCal = int(k * (x_max - x_min))
                    imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                    wGap = (imgsize - wCal) // 2
                    imgWhite[:, wGap:wGap + wCal] = imgResize
                else:
                    k = imgsize / (x_max - x_min)
                    hCal = int(k * (y_max - y_min))
                    imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                    hGap = (imgsize - hCal) // 2
                    imgWhite[hGap:hGap + hCal, :] = imgResize

                # Classify two-handed gesture
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                detected_word = labels[index]
                detected_texts.append(detected_word)

        else:  # Single-hand detection
            for hand in hands:
                x, y, w, h = hand['bbox']
                x1, y1 = max(0, x - offset), max(0, y - offset)
                x2, y2 = min(image.shape[1], x + w + offset), min(image.shape[0], y + h + offset)
                imgCrop = image[y1:y2, x1:x2]

                if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                    aspectRatio = h / w
                    imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

                    if aspectRatio > 1:
                        k = imgsize / h
                        wCal = int(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                        wGap = (imgsize - wCal) // 2
                        imgWhite[:, wGap:wGap + wCal] = imgResize
                    else:
                        k = imgsize / w
                        hCal = int(k * h)
                        imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                        hGap = (imgsize - hCal) // 2
                        imgWhite[hGap:hGap + hCal, :] = imgResize

                    # Classify single-hand gesture
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    detected_word = labels[index]
                    detected_texts.append(detected_word)

        # Update detected words
        if detected_texts:
            detected_word = detected_texts[0]  # Take first detected word
            st.session_state["detected_words"] = detected_word

            # Speak only if new word is detected
            if detected_word != st.session_state["last_spoken_word"]:
                threading.Thread(target=speak, args=(detected_word,), daemon=True).start()
                st.session_state["last_spoken_word"] = detected_word

        # Update sentence display
        sentence_display.write(f"### {st.session_state['detected_words']}")

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(imgOutput, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display video in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()
    FRAME_WINDOW.empty()
