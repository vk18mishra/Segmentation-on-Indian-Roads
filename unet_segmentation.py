
import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd
import altair as alt
import pandas as pd
import numpy as np
import os
import urllib
import cv2


def main():
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "See Default Video", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code("preprocessing_&_unet_training")
    elif app_mode == "See Default Video":
        readme_text.empty()
        see_default_video()

def download_file(file_path):

    if file_path=='default_video':
        gdd.download_file_from_google_drive(file_id=file_path['url'],
                                            dest_path='./default_video.mp4')
    if file_path=='unet_weights':
        gdd.download_file_from_google_drive(file_id=file_path['url'],
                                            dest_path='./unet_weights.mp4')

def see_default_video():
    video_file = open('default_video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


EXTERNAL_DEPENDENCIES = {
    "default_video": {
        "url": "13bjrQ7S2z-pyV2-zfeqEu4-RwRCO1uxq"
    },
    "unet_weights": {
        "url": "1Ot_RHbvD_mxb6T8jzOqk2J3uXBiRJtD7"
    }
}

if __name__ == "__main__":
    main()