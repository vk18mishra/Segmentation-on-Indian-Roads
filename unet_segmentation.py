import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd
import urllib
from PIL import Image

def main():
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    st.sidebar.title("What to do")
    app_mode_main = st.sidebar.selectbox("Choose The Application Mode",
                                    ["Show instructions",
                                     "Sample Video", "Sample Output Video",
                                     "Preprocessing source code", "App Source code",
                                     "Model-UNET", "Model-CANET"])
    if app_mode_main == "Show instructions":
        st.sidebar.success('Choose the preferred option from the dropdown')
    elif app_mode_main == "Preprocessing source code":
        readme_text.empty()
        st.title('Preprocessing source code')
        st.code(get_file_content_as_string("preprocessing.py"))
    elif app_mode_main == "App Source code":
        readme_text.empty()
        st.title('App Source code')
        st.code(get_file_content_as_string("unet_segmentation.py"))
    elif app_mode_main == "Sample Video":
        readme_text.empty()
        see_default_video()
    elif app_mode_main == "Sample Output Video":
        readme_text.empty()
        see_sample_output()
    elif app_mode_main == "Model-UNET":
        readme_text.empty()
        app_mode_unet = st.sidebar.selectbox("Segmentation Using UNET",
                                        ["Complete UNET Architecture",
                                         "Calculate Segmentation Maps on User Input Video using UNET",
                                         "UNET source code"])
        if app_mode_unet == "UNET source code":
            readme_text.empty()
            st.title('UNET source code')
            st.code(get_file_content_as_string("UNET%20Segmentation.py"))
        elif app_mode_unet == "Complete UNET Architecture":
            readme_text.empty()
            st.title('UNET Architecture')
            image = Image.open('unet_architecture.jpeg')
            st.image(image, caption='Complete UNET Architecture',
                     use_column_width=True)
        elif app_mode_unet == "Calculate Segmentation Maps on User Input Video using UNET":
            readme_text.empty()
            st.title('Calculate Segmentation Maps on User Input Video using UNET')
            st.title('Coming Soon...')
            st.write('Memory Constraint over Free Tier subscriptions of Cloud Providers is the Issue.')
    elif app_mode_main == "Model-CANET":
        readme_text.empty()
        app_mode_canet = st.sidebar.selectbox("Segmentation Using CANET",
                                              ["Complete CANET Architecture",
                                               "Complete CANET Tensorflow Model",
                                               "Calculate Segmentation Maps on User Input Video using CANET",
                                               "CANET source code"])

        if app_mode_canet == "CANET source code":
            readme_text.empty()
            st.title('CANET source code')
            st.code(get_file_content_as_string("canet_segmentation.py"))
        elif app_mode_canet == "Complete CANET Tensorflow Model":
            readme_text.empty()
            st.title('Complete CANET Tensorflow Model')
            image = Image.open('canet_main.jpeg')
            st.image(image, caption='Complete CANET Tensorflow Model',
                     use_column_width=True)
        elif app_mode_canet == "Complete CANET Architecture":
            readme_text.empty()
            st.title('Attention-guided Chained Context Aggregation for Semantic Segmentation')
            st.title('Complete CANET Architecture')
            image = Image.open('canet_description.jpeg')
            st.image(image, caption='Complete CANET Architecture',
                     use_column_width=True)
            st.title('Convolution Block')
            image = Image.open('convolution_block.jpeg')
            st.image(image, caption='Convolution Block',
                     use_column_width=True)
            st.title('Identity Block')
            image = Image.open('identity_block.jpeg')
            st.image(image, caption='Identity Block',
                     use_column_width=True)
            st.title('CAM Module')
            image = Image.open('CAM_module.jpeg')
            st.image(image, caption='CAM Module',
                     use_column_width=True)
            st.title('AGCN Block')
            image = Image.open('AGCN_block.jpeg')
            st.image(image, caption='AGCN Block',
                     use_column_width=True)
        elif app_mode_canet == "Calculate Segmentation Maps on User Input Video using CANET":
            readme_text.empty()
            st.title('Calculate Segmentation Maps on User Input Video using CANET')
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
    if file_path == 'canet_main':
        gdd.download_file_from_google_drive(file_id='1ZKm2h3yWNmFAWpyyBXF48KG9gfLV-u5W',
                                        dest_path='./canet_main.jpeg', showsize=True)
    if file_path == 'canet_description':
        gdd.download_file_from_google_drive(file_id='1DYDe4QzvtIEkwgfjgrWXky0uhk0Lw0SO',
                                        dest_path='./canet_description.jpeg', showsize=True)
    if file_path == 'convolution_block':
        gdd.download_file_from_google_drive(file_id='1q3OWrEKrTOCvbX0v8nfsdWREP9wIEhC3',
                                        dest_path='./convolution_block.jpeg', showsize=True)
    if file_path == 'identity_block':
        gdd.download_file_from_google_drive(file_id='1Dk7dqYXjmOABKBcJQrIJBcGQCzmP3HF5',
                                        dest_path='./identity_block.jpeg', showsize=True)
    if file_path == 'CAM_module':
        gdd.download_file_from_google_drive(file_id='1D9pD8Rl0OV_F_9WwYB0_6Gx5mv3lHuJQ',
                                        dest_path='./CAM_module.jpeg', showsize=True)
    if file_path == 'AGCN_block':
        gdd.download_file_from_google_drive(file_id='1KN_S7obg0EMIo2YglE10gNaM4S2UU-sz',
                                        dest_path='./AGCN_block.jpeg', showsize=True)

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
    },
    "canet_main": {
        "url": "1ZKm2h3yWNmFAWpyyBXF48KG9gfLV-u5W"
    },
    "canet_description": {
        "url": "1DYDe4QzvtIEkwgfjgrWXky0uhk0Lw0SO"
    },
    "convolution_block": {
        "url": "1q3OWrEKrTOCvbX0v8nfsdWREP9wIEhC3"
    },
    "identity_block": {
        "url": "1Dk7dqYXjmOABKBcJQrIJBcGQCzmP3HF5"
    },
    "CAM_module": {
        "url": "1D9pD8Rl0OV_F_9WwYB0_6Gx5mv3lHuJQ"
    },
    "AGCN_block": {
        "url": "1KN_S7obg0EMIo2YglE10gNaM4S2UU-sz"
    }
}

if __name__ == "__main__":
    main()