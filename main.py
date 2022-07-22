import streamlit as st
import numpy as np
import cv2

# Load image.
uploaded_file = st.sidebar.file_uploader("Upload an image:")

if uploaded_file is not None:
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # convert string data to numpy array
    npimg = np.fromstring(uploaded_file.getvalue(), np.uint8)
    # convert numpy array to image
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Switch to HSV for simplier color handling.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower_h, upper_h = st.slider("Select a hue range:", 0, 360, (0,360))
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    # Loop on all objects we want to detect.
    if faces == ():
        print("No faces found")

    # We iterate through our faces array and draw a rectangle
    # over each face in faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image)