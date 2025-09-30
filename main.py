import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np


st.set_page_config(layout="wide")
col_1, col_2 = st.columns([2, 3])

# # --- Session state ---
# if "ai_queue" not in st.session_state:
#     st.session_state["ai_queue"] = Queue()
# if "out_text" not in st.session_state:
#     st.session_state["out_text"] = ""


prev_pos = None
canvas = None
out_text = None

# --- AI setup ---
# 

# ---------------- Fungsi ----------------
def getHandInfo(img, detector):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(img, info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None

    if fingers == [0, 1, 0, 0, 0]:  # telunjuk
        current_pos = tuple(map(int, lmList[8][0:2]))
        if prev_pos is not None:
            cv2.line(canvas, current_pos, prev_pos, (0, 0, 100), 7)

    elif fingers == [1, 1, 1, 1, 1]:  # reset
        canvas = np.zeros_like(img)

    return current_pos, canvas, fingers

def sendToAI(model, canvas, fingers, input_text):
    if fingers == [1, 1, 1, 1, 0]:  # trigger
        image = Image.fromarray(canvas)
        response = model.generate_content([input_text, image])
        return response.text
    return None

# ---------------- UI ----------------
with col_1:
    input_text = st.text_input("Masukkan perintah")
    st.subheader("Jawaban")
    output_text_area = st.empty()

with col_2:
    detector = HandDetector(
        staticMode=False, maxHands=1, modelComplexity=1,
        detectionCon=0.5, minTrackCon=0.5
    )

    def callback(frame):
        global prev_pos, canvas, out_text
        img = frame.to_ndarray(format="bgr24")

        if canvas is None:
            canvas = np.zeros_like(img)

        info = getHandInfo(img, detector)
        if info:
            new_pos, canvas, fingers = draw(img, info, prev_pos, canvas)
            prev_pos = new_pos


        image_combined = cv2.addWeighted(img, 0.7, canvas, 1, 0)
        return av.VideoFrame.from_ndarray(image_combined, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )


