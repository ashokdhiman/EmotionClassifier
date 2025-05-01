# Import required libraries
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Define emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Load the pre-trained model
@st.cache_resource
def load_emotion_model():
    model = load_model('model_80_65.h5', compile=False)
    model.load_weights('model_weights_80_65.weights.h5')
    return model


classifier = load_emotion_model()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Define the Video Transformer
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.output_text = "Neutral"

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
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


# Main function
def main():
    st.title("Real Time Face Emotion Detection Application 😠🤮😨😀😐😔😮")

    activities = ["Home", "Live Face Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown(""" Developed by Ashok and Dharna""")

    if choice == "Home":
        html_temp_home = """
        <div style="background-color:#FC4C02;padding:10px">
            <h2 style="color:white;text-align:center;">Start Your Real Time Face Emotion Detection</h2>
        </div><br>"""
        st.markdown(html_temp_home, unsafe_allow_html=True)
        st.write("""
        Ever wondered if your computer can recognize your emotions?
        Let's find out!

        1. Select "Live Face Emotion Detection" from the dropdown.
        2. Allow webcam access.
        3. Watch the model detect your emotions in real-time!
        """)

    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed")
        st.subheader("Show your emotions to the camera! 📷")
        st.write("Click 'Start' to open the webcam and detect emotions in real-time.")
        webrtc_streamer(
            key="emotion_detection",
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
        )

    elif choice == "About":
        html_temp_about = """
        <div style="background-color:#36454F;padding:10px">
            <h4 style="color:white;">
            This app predicts facial emotions using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.
            Face detection is achieved through OpenCV.
            </h4>
        </div><br>"""
        st.markdown(html_temp_about, unsafe_allow_html=True)
        st.write("""
        **Technologies Used:**
        - TensorFlow & Keras for deep learning
        - OpenCV for face detection
        - Streamlit & Streamlit-WebRTC for building the real-time web app
        """)


if __name__ == "__main__":
    main()
