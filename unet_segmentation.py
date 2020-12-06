import segmentation_models as sm
import numpy as np
import streamlit as st
from google_drive_downloader import GoogleDriveDownloader as gdd
import urllib
import cv2
import tensorflow as tf
import keras
from moviepy.editor import VideoFileClip, clips_array

class tqdm:
    def __init__(self, iterable, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)


def main():
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    with st.spinner('UNet Model is being Loaded...'):
        model = load_umodel()
    st.success('Model has been Loaded into Memory')

    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "See Default Video", "See Sample Output Video", "Run UNet Model and See Output", "Show the UNet source code", "Show the Final source code"])
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
    elif app_mode == "See Sample Output Video":
        readme_text.empty()
        see_sample_output()
    elif app_mode == "Run UNet Model and See Output":
        readme_text.empty()
        st.title('Running the Trained Unet model on the Input')
        images = get_frames()

        tot_frames = len(images)
        pred_mask = []
        size = (256, 256)
        for i in tqdm(range(tot_frames), title='Calculating Masks from Trained UNet...'):
            # Original Image
            image = cv2.resize(images[i], size, interpolation=cv2.INTER_NEAREST)

            # Predicted Segmentation Map
            pred_mask_i = model.predict(image[np.newaxis, :, :, :])
            pred_mask_i = tf.argmax(pred_mask_i, axis=-1)
            tmp_np = np.array(pred_mask_i)
            tmp_np = np.rollaxis(tmp_np, 0, 3)
            image_1_0 = cv2.resize(tmp_np, size, interpolation=cv2.INTER_NEAREST)
            pred_mask.append(image_1_0)

        run_unet_see_output(images, pred_mask)

def download_file(file_path):

    if file_path == 'default_video':
        with st.spinner('Downloading Default Video(35MB), Please Wait....'):
            gdd.download_file_from_google_drive(file_id='13bjrQ7S2z-pyV2-zfeqEu4-RwRCO1uxq',
                                                dest_path='./default_video.mp4', showsize=True)
        st.success('Default Video Downloaded!')
    if file_path == 'sample_output':
        with st.spinner('Downloading Sample Output Video(18MB), Please Wait....'):
            gdd.download_file_from_google_drive(file_id='1Ar039WQSsgKAGK-3m5H8x5kI-rhxU_kU',
                                                dest_path='./sample_output.mp4', showsize=True)
        st.success('Sample Output Video Downloaded!')
    if file_path == 'unet_full':
        with st.spinner('Downloading Trained Unet Model(280MB), Please Wait....'):
            gdd.download_file_from_google_drive(file_id='18wYH91CgrevmIEdAoylTb6mdkW9LS9Jy',
                                                dest_path='./unet_full.h5', showsize=True)
        st.success('Trained Unet Model Downloaded!')

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

@st.cache
def load_umodel():
    return keras.models.load_model('unet_full.h5')

def get_frames():
    cap = cv2.VideoCapture("default_video.mp4")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (256, 256)
    vidcap = cv2.VideoCapture('default_video.mp4')
    success, image = vidcap.read()
    count = 0
    images = []
    with st.spinner('Breaking the Video into Individual Frames, Please Wait....'):
        for i in tqdm(range(length), title='Breaking Vid into Frames'):
            image_1 = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
            images.append(image_1)
            success, image = vidcap.read()
            count = count + 1
    st.success('We have Individual Frames Now!')
    return images

def run_unet_see_output(images, pred_mask):
    # st.title('Applying the Trained Unet model on the Input')
    # st.write('Loading the Model, Please Wait...')
    # with st.spinner('Loading the Trained UNet Model into Memory, Please Wait....'):
    # model = load_model()
    # st.write('UNet is loaded!')
    size = (256, 256)
    # Original(Normal) Video
    with st.spinner('Combining the simple frames back into a video, Please Wait....'):
        out = cv2.VideoWriter('original_video.mp4', cv2.VideoWriter_fourcc(*'X264'), 15, size)
        for i in tqdm(range(len(images)), title='Building Original Video...'):
            out.write(images[i])
        out.release()
    st.success('We have the original Video now!')
    # Masked Video
    with st.spinner('Combining the segmented frames into a video, Please Wait....'):
        out1 = cv2.VideoWriter('masked_video.mp4', cv2.VideoWriter_fourcc(*'X264'), 15, size, 0)
        for i in tqdm(range(len(pred_mask)), title='Building Video from Masks...'):
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
    },
    "sample_output": {
        "url": "1Ar039WQSsgKAGK-3m5H8x5kI-rhxU_kU"
    }
}

if __name__ == "__main__":
    main()