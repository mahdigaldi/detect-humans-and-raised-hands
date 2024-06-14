import cv2
import numpy as np
import mediapipe as mp
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# Initialize Mediapipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load YOLO model
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Function to detect humans and raised hands
def detect_humans_and_hands(frame):
    height, width, channels = frame.shape

    # Detecting objects with YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    human_count = 0
    raised_hands_count = 0

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class 'person'
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    human_count = len(indexes)

    # Mediapipe Pose detection
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(img_rgb)

    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            if landmark.visibility > 0.5 and landmark.y < pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y:
                raised_hands_count += 1

    # Mediapipe Hands detection
    hands_results = hands.process(img_rgb)

    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                if landmark.y < hands_results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y:
                    raised_hands_count += 1
                    break

    return human_count, raised_hands_count

# Function to update GUI
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    human_count, raised_hands_count = detect_humans_and_hands(frame)

    # Update text boxes
    human_count_var.set(str(human_count))
    raised_hands_count_var.set(str(raised_hands_count))

    # Display image in PictureBox
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    root.after(10, update_frame)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize GUI
root = Tk()
root.title("Human and Raised Hands Detection")

main_frame = Frame(root)
main_frame.pack()

panel = Label(main_frame)
panel.pack()

label_human_count = Label(main_frame, text="Number of Humans:")
label_human_count.pack()
human_count_var = StringVar()
human_count_textbox = Entry(main_frame, textvariable=human_count_var, state='readonly')
human_count_textbox.pack()

label_raised_hands_count = Label(main_frame, text="Number of Raised Hands:")
label_raised_hands_count.pack()
raised_hands_count_var = StringVar()
raised_hands_count_textbox = Entry(main_frame, textvariable=raised_hands_count_var, state='readonly')
raised_hands_count_textbox.pack()

root.after(10, update_frame)
root.mainloop()

# Release webcam
cap.release()
cv2.destroyAllWindows()