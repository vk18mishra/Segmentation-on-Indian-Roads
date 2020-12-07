import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd
import urllib
from PIL import Image

def main():
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Complete UNet Architecture",
                                     "Sample Video", "Sample Output Video",
                                     "Calculate Segmentation Maps on User given Video using UNet",
                                     "UNet source code", "App Source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue "Select Something from DropDown!".')
    elif app_mode == "UNet source code":
        readme_text.empty()
        st.code(get_file_content_as_string("preprocessing_unet_training.py"))
    elif app_mode == "App Source code":
        readme_text.empty()
        st.code(get_file_content_as_string("unet_segmentation.py"))
    elif app_mode == "Sample Video":
        readme_text.empty()
        see_default_video()
    elif app_mode == "Complete UNet Architecture":
        readme_text.empty()
        st.title('UNet Architecture')
        image = Image.open('unet_architecture.jpeg')
        st.image(image, caption='Complete UNet Architecture',
                 use_column_width=True)
    elif app_mode == "Sample Output Video":
        readme_text.empty()
        see_sample_output()
    elif app_mode == "Calculate Segmentation Maps on User given Video using UNet":
        readme_text.empty()
        st.title('Coming Soon...')
        st.write('Memory Constraint over Free Tier subscriptions of Cloud Providers is the Issue.')

def download_file(file_path):

    if file_path == 'default_video':
        gdd.download_file_from_google_drive(file_id='13bjrQ7S2z-pyV2-zfeqEu4-RwRCO1uxq',
                                        dest_path='./default_video.mp4', showsize=True)
    if file_path == 'sample_output':
        gdd.download_file_from_google_drive(file_id='1Ar039WQSsgKAGK-3m5H8x5kI-rhxU_kU',
                                        dest_path='./sample_output.mp4', showsize=True)
    if file_path == 'unet_arch':
        gdd.download_file_from_google_drive(file_id='1uhmVB6choPnu8y6ZZDPE_VRtKCtKar2a',
                                        dest_path='./unet_architecture.jpeg', showsize=True)

def see_default_video():
    st.title('Displaying the Default Video')
    video_file = open('default_video.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)

def see_sample_output():
    st.title('Displaying the Sample Output Video')
    video_file = open('sample_output.mp4', 'rb')
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
    "sample_output": {
        "url": "1Ar039WQSsgKAGK-3m5H8x5kI-rhxU_kU"
    },
    "unet_arch": {
        "url": "1uhmVB6choPnu8y6ZZDPE_VRtKCtKar2a"
    }
}

if __name__ == "__main__":
    main()
