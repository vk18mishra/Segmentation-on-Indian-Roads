# pip install numpy==1.19.3
# pip install tensorflow==2.2.0
# pip install keras==2.3.1
# pip install -U segmentation-models==0.2.1
# pip install moviepy

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd
import urllib
import cv2
import tensorflow as tf
import keras
from moviepy.editor import VideoFileClip, clips_array


def main():
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "See Default Video", "Run UNet Model and See Output", "Show the UNet source code", "Show the Final source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue "Select Something from DropDown!".')
    elif app_mode == "Show the UNet source code":
        readme_text.empty()
        st.code(get_file_content_as_string("preprocessing_unet_training.py"))
    elif app_mode == "Show the Final source code":
        readme_text.empty()
        st.code(get_file_content_as_string("unet_segmentation.py"))
    elif app_mode == "See Default Video":
        readme_text.empty()
        see_default_video()
    elif app_mode == "Run UNet Model and See Output":
        readme_text.empty()
        run_unet_see_output()

def download_file(file_path):

    if file_path=='default_video':
        with st.spinner('Downloading Default Video(35MB), Please Wait....'):
            gdd.download_file_from_google_drive(file_id='13bjrQ7S2z-pyV2-zfeqEu4-RwRCO1uxq',
                                                dest_path='./default_video.mp4', showsize=True)
        st.success('Default Video Downloaded!')

    if file_path=='unet_full':
        with st.spinner('Downloading Trained Unet Model(280MB), Please Wait....'):
            gdd.download_file_from_google_drive(file_id='18wYH91CgrevmIEdAoylTb6mdkW9LS9Jy',
                                                dest_path='./unet_full.h5', showsize=True)
        st.success('Trained Unet Model Downloaded!')

def see_default_video():
    st.title('Displaying the Default Video')
    video_file = open('default_video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def run_unet_see_output():
    st.title('Applying the Trained Unet model on the Input')
    size = (256, 256)
    vidcap = cv2.VideoCapture('default_video.mp4')
    success, image = vidcap.read()
    count = 0
    images = []
    with st.spinner('Breaking the Video into Individual Frames, Please Wait....'):
        while success:
            image_1 = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
            images.append(image_1)
            success, image = vidcap.read()
            count = count + 1
    st.success('We have Individual Frames Now!')
    with st.spinner('Loading the Trained UNet Model into Memory, Please Wait....'):
        model = keras.models.load_model('unet_full.h5')
    st.success('UNet is loaded!')
    tot_frames = len(images)
    pred_mask = []
    with st.spinner('Predicting the segmentation maps of individual frames by passing them into the model, '
                    'Please Wait....'):
        for i in range(tot_frames):
            # Original Image
            image = cv2.resize(images[i], size, interpolation=cv2.INTER_NEAREST)

            # Predicted Segmentation Map
            pred_mask_i = model.predict(image[np.newaxis, :, :, :])
            pred_mask_i = tf.argmax(pred_mask_i, axis=-1)
            tmp_np = np.array(pred_mask_i)
            tmp_np = np.rollaxis(tmp_np, 0, 3)
            image_1_0 = cv2.resize(tmp_np, size, interpolation=cv2.INTER_NEAREST)
            pred_mask.append(image_1_0)
    st.success('We have respective segmentation maps now!')
    # Original(Normal) Video
    with st.spinner('Combining the simple frames back into a video, Please Wait....'):
        out = cv2.VideoWriter('original_video.mp4', cv2.VideoWriter_fourcc(*'X264'), 15, size)
        for i in range(len(images)):
            out.write(images[i])
        out.release()
    st.success('We have the original Video now!')
    # Masked Video
    with st.spinner('Combining the segmented frames into a video, Please Wait....'):
        out1 = cv2.VideoWriter('masked_video.mp4', cv2.VideoWriter_fourcc(*'X264'), 15, size, 0)
        for i in range(len(pred_mask)):
            frame = pred_mask[i]
            frame = np.uint8(10 * frame)
            out1.write(frame)
        out1.release()
    st.success('We have the segmented Video now!')
    with st.spinner('Merging both the videos side by side, Please Wait....'):
        clip1 = VideoFileClip("original_video.mp4").margin(10)  # add 10px contour
        clip2 = VideoFileClip("masked_video.mp4")
        final_clip = clips_array([[clip1, clip2]])
        final_clip.resize(width=480).write_videofile("my_output_final.mp4")
    st.success('We have the Final Video now!')
    st.header('Displaying the Final Output Video:')
    video_file = open('my_output_final.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/vk18mishra/Segmentation-on-Indian-Roads/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


EXTERNAL_DEPENDENCIES = {
    "default_video": {
        "url": "13bjrQ7S2z-pyV2-zfeqEu4-RwRCO1uxq"
    },
    "unet_full": {
        "url": "18wYH91CgrevmIEdAoylTb6mdkW9LS9Jy"
    }
}

if __name__ == "__main__":
    main()