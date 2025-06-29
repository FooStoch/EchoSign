import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import pickle
import numpy as np
import mediapipe as mp

# Page configuration
st.set_page_config(page_title="EchoSign", layout="centered")
st.title("EchoSign")

# Load model once
@st.cache_resource
def get_model():
    model_dict = pickle.load(open("./model.p", "rb"))
    return model_dict['model']

model = get_model()

# Initialize Mediapipe hands (video mode)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)

labels_dict = {i: chr(ord('A') + i) for i in range(26)}

class VideoTransformer(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        H, W, _ = img.shape
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                data = []
                for x, y in zip(xs, ys):
                    data.append(x - min(xs))
                    data.append(y - min(ys))

                x1 = int(min(xs) * W) - 10
                y1 = int(min(ys) * H) - 10
                x2 = int(max(xs) * W) + 10
                y2 = int(max(ys) * H) + 10

                char = ""
                if len(data) == 42:
                    pred = model.predict([np.asarray(data)])[0]
                    char = labels_dict[int(pred)]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    img,
                    char,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA
                )
        # Return processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Control
if st.checkbox("Open Camera"):
    webrtc_streamer(
        key="echo-sign",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
else:
    st.write("Check the box to start the camera.")
