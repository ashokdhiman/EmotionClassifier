import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


@st.cache_resource
def load_emotion_model():
    model = load_model('model_80_65.h5', compile=False)
    model.load_weights('model_weights_80_65.weights.h5')
    return model


classifier = load_emotion_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.output_text = "Neutral"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if roi_gray is not None and np.sum(roi_gray) != 0:
                roi = roi_gray.astype('float32') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi, verbose=0)[0]
                maxindex = int(np.argmax(prediction))
                self.output_text = emotion_labels[maxindex]

            label_position = (x, y - 10)
            cv2.putText(img, self.output_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img


def main():

    st.title("Real Time Face Emotion Detection Application 😠🤮😨😀😐😔😮")

    with st.sidebar:
        st.markdown("## 📋 Navigation")
        activities = ["Home", "Live Face Emotion Detection", "About"]
        choice = st.selectbox("Select Activity", activities)
        st.markdown("---")
        st.markdown("👨‍💻 *Developed by Ashok and Dharna*")

    if choice == "Home":
        st.markdown(
            """
            <div style="background-color:#FF6F61;padding:16px;border-radius:10px;">
                <h3 style="color:white;text-align:center;">Real time facial emotions classifier</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("""
        ### 🤔 What's this app for?
        This app uses your webcam to analyze your facial expressions and predict your emotion in real time.

        #### 🔧 How to use:
        - Go to **Live Face Emotion Detection** from the sidebar.
        - Grant webcam permission.
        - Watch your emotions being detected live on the screen!
        """)

    elif choice == "Live Face Emotion Detection":
        st.markdown(
            """
            <div style="background-color:#4682B4;padding:16px;border-radius:10px;">
                <h3 style="color:white;text-align:center;">🎥 Live Webcam Emotion Detection</h3>
            </div><br>
            """, unsafe_allow_html=True
        )
        st.write("Press 'Start' to launch your webcam and see your live facial emotion predictions.")
        webrtc_streamer(
            key="emotion_detection",
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
        )

    elif choice == "About":
        st.markdown(
            """
            <div style="background-color:#2E8B57;padding:16px;border-radius:10px;">
                <h3 style="color:white;">📚 About This App</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("""
        This project demonstrates facial emotion recognition using deep learning.

        #### 🛠 Technologies Used:
        - **TensorFlow/Keras** for emotion prediction
        - **OpenCV** for face detection
        - **Streamlit & Streamlit-WebRTC** for real-time webcam interface

        The model detects seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
        """)


if __name__ == "__main__":
    main()