import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

 # Import Library
import cvzone
from cvzone.HandTrackingModule import HandDetector
import  numpy as np
import google.generativeai as genai
from PIL import Image

st.set_page_config(layout="wide")
col_1, col_2 = st.columns([2,3])

with col_1:
    input_text = st.text_input(label="Masukkan perintah")
    st.subheader("Jawaban")
    output_text_area = st.text("   ")


with col_2:
    # Buat detector sekali saja, jangan di dalam callback
    detector = HandDetector(
        staticMode=False, 
        maxHands=1, 
        modelComplexity=1, 
        detectionCon=0.5, 
        minTrackCon=0.5
    )

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")

        hands, img = detector.findHands(img, draw=True, flipType=True)

        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  
            bbox1 = hand1["bbox"]  
            center1 = hand1['center']  
            handType1 = hand1["type"]  

            # Contoh update text di kolom kiri
            output_text_area.text(f"Terdeteksi {handType1} hand di {center1}")

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                {"urls": ["stun:stun.cloudflare.com:3478"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]},
                {"urls": ["stun:openrelay.metered.ca:80"]},
            ]
        }
    )
 
