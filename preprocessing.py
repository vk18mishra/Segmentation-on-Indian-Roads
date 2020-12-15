# -*- coding: utf-8 -*-
"""Preprocessing.ipynb

# Segmentation of Indian Traffic
"""

!pip install tensorflow==2.2.0
!pip install keras==2.3.1

import math
from PIL import Image, ImageDraw
from PIL import ImagePath
import pandas as pd
import os
from os import path
from tqdm import tqdm
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib

"""# Task 1: Preprocessing"""

#Loading Data through Google Drive
# Install the PyDrive wrapper & import libraries.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_id = '1uFqgly3XZefQlE8OXUl5CUei3Mh7yhmd'
downloaded = drive.CreateFile({'id':file_id})
downloaded.FetchMetadata(fetch_all=True)
downloaded.GetContentFile(downloaded.metadata['title'])

#Loading data.zip through curlwget
# !wget --header="Host: doc-00-90-docs.googleusercontent.com" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: en-US,en;q=0.9,es;q=0.8" --header="Cookie: AUTH_j9bcrmgt4omen2e6cvbm2ejader83m0d_nonce=0htrs50lkrigm" --header="Connection: keep-alive" "https://doc-00-90-docs.googleusercontent.com/docs/securesc/gav8itevci1g4kd0q1cuucl5an9d430s/npkcu2rijhd5bpm1vr3997fvodsafjhg/1605533550000/00484516897554883881/05016653509110039968/1iQ93IWVdR6dZ6W7RahbLq166u-6ADelJ?e=download&authuser=0&nonce=0htrs50lkrigm&user=05016653509110039968&hash=a6i4pesktp8bmu2ol5ko0gjih4i3kn5o" -c -O 'data.zip'

!unzip 'data.zip'
#print("Data is Loaded")

"""## 1. Get all the file name and corresponding json files"""

frames_list = os.listdir('data/images')
images_file = []
for i in frames_list:
  dir_img = 'data/images/'+i
  dir_mask = 'data/mask/'+i
  directory = os.fsencode(dir_img)
  for file in (os.listdir(directory)):
      filename = os.fsdecode(file)
      if filename.endswith(".jpg"):
        tmp = dir_img+'/'+filename
        images_file.append(tmp)
len(images_file)

masks_file = []
for i in images_file:
  file_chk1 = i.split(sep='_')[0]
  file_chk = 'data/mask/'+file_chk1.split(sep='/')[2]
  file_mask_chk = 'data/mask/'+file_chk.split(sep='/')[2]+'/'+file_chk1.split(sep='/')[3]
  frame_chk = file_chk1.split(sep='/')[3]
  directory = os.fsencode(file_chk)
  for file in (os.listdir(directory)):
      filename = os.fsdecode(file)
      mask_frame_chk = filename.split(sep='_')[0]
      if mask_frame_chk==frame_chk:
        tmp1 = filename.split(sep='_')[1]
        tmp = file_mask_chk+'_'+tmp1+'_'+filename.split(sep='_')[2]
        masks_file.append(tmp)
len(masks_file)

def return_file_names_df(root_dir):
    # write the code that will create a dataframe with two columns ['images', 'json']
    # the column 'image' will have path to images
    # the column 'json' will have path to json files
    data_df = pd.DataFrame()
    data_df['image'] = images_file
    data_df['json'] = masks_file
    return data_df

root_dir = 'data'
data_df = return_file_names_df(root_dir)
data_df.head()

"""> If you observe the dataframe, we can consider each row as single data point, where first feature is image and the second feature is corresponding json file"""

def grader_1(data_df):
    for i in data_df.values:
        if not (path.isfile(i[0]) and path.isfile(i[1]) and i[0][12:i[0].find('_')]==i[1][10:i[1].find('_')]):
            return False
    return True

grader_1(data_df)

data_df.shape

"""## 2. Structure of sample Json file

#### Compute the unique labels

Let's see how many unique objects are there in the json file.
to see how to get the object from the json file please check <a href='https://www.geeksforgeeks.org/read-json-file-using-python/'>this blog </a>
"""

def return_unique_labels(data_df):
    # for each file in the column json
    #       read and store all the objects present in that file
    # compute the unique objects and retrun them
    # if open any json file using any editor you will get better sense of it
    labels = set()
    label_k = 0
    for filename in data_df['json']:
      with open(filename) as f:
        data = json.load(f)
        #data_new = json.dumps(data, indent=2)
        for i in data['objects']:
          if i['label'] not in labels:
            label_k = label_k + 1
            labels.add(i['label'])
    #label_k, labels
    return list(labels)

unique_labels = return_unique_labels(data_df)

len(unique_labels)

label_clr = {'road':10, 'parking':20, 'drivable fallback':20,'sidewalk':30,'non-drivable fallback':40,'rail track':40,\
                        'person':50, 'animal':50, 'rider':60, 'motorcycle':70, 'bicycle':70, 'autorickshaw':80,\
                        'car':80, 'truck':90, 'bus':90, 'vehicle fallback':90, 'trailer':90, 'caravan':90,\
                        'curb':100, 'wall':100, 'fence':110,'guard rail':110, 'billboard':120,'traffic sign':120,\
                        'traffic light':120, 'pole':130, 'polegroup':130, 'obs-str-bar-fallback':130,'building':140,\
                        'bridge':140,'tunnel':140, 'vegetation':150, 'sky':160, 'fallback background':160,'unlabeled':0,\
                        'out of roi':0, 'ego vehicle':170, 'ground':180,'rectification border':190,\
                   'train':200}

def grader_2(unique_labels):
    if (not (set(label_clr.keys())-set(unique_labels))) and len(unique_labels) == 40:
        print("True")
    else:
        print("Flase")

grader_2(unique_labels)

list_classes = [label_clr[key] for key in label_clr.keys()]
list_classes = list(set(list_classes))
list_classes = [int(cvf/10) for cvf in list_classes]
list_classes.sort()
print(list_classes, '\n',len(list_classes))

"""<pre>
* here we have given a number for each of object types, if you see we are having 21 different set of objects
* Note that we have multiplies each object's number with 10, that is just to make different objects look differently in the segmentation map
* Before you pass it to the models, you might need to devide the image array /10.
</pre>

## 3. Extracting the polygons from the json files
"""

def get_poly(file):
    # this function will take a file name as argument
    
    # it will process all the objects in that file and returns
    
    # label: a list of labels for all the objects label[i] will have the corresponding vertices in vertexlist[i]
    # len(label) == number of objects in the image
    
    # vertexlist: it should be list of list of vertices in tuple formate 
    # ex: [[(x11,y11), (x12,y12), (x13,y13) .. (x1n,y1n)]
    #     [(x21,y21), (x22,y12), (x23,y23) .. (x2n,y2n)]
    #      .....
    #     [(xm1,ym1), (xm2,ym2), (xm3,ym3) .. (xmn,ymn)]]
    # len(vertexlist) == number of objects in the image
    
    # * note that label[i] and vertextlist[i] are corresponds to the same object, one represents the type of the object
    # the other represents the location
    
    # width of the image
    # height of the image
    with open(file) as file:
      label = []
      vertexlist = []
      data = json.load(file)
      h = data['imgHeight']
      w = data['imgWidth']
      for i in data['objects']:
        label.append(i['label'])
        list_vertices = []
        for j in i['polygon']:
          list_vertices.append(tuple(j))
        vertexlist.append(list_vertices)
    return w, h, label, vertexlist

def grader_3(file):
    w, h, labels, vertexlist = get_poly(file)
    print(len((set(labels)))==18 and len(vertexlist)==227 and w==1920 and h==1080 \
          and isinstance(vertexlist,list) and isinstance(vertexlist[0],list) and isinstance(vertexlist[0][0],tuple) )

grader_3('data/mask/201/frame0029_gtFine_polygons.json')

"""## 4. Creating Image segmentations by drawing set of polygons"""

def compute_masks(data_df):
    # after you have computed the vertexlist plot that polygone in image like this
    
    # img = Image.new("RGB", (w, h))
    # img1 = ImageDraw.Draw(img)
    # img1.polygon(vertexlist[i], fill = label_clr[label[i]])
    
    # after drawing all the polygons that we collected from json file, 
    # you need to store that image in the folder like this "data/output/scene/framenumber_gtFine_polygons.png"
    
    # after saving the image into disk, store the path in a list
    # after storing all the paths, add a column to the data_df['mask'] ex: data_df['mask']= mask_paths
    #try:
    masks_all = []
    for i in tqdm(range(len(data_df['json']))):
      file_img = data_df['image'][i]
      file_json = (data_df['json'][i])
      file_mask = file_json.replace('/mask/', '/output/')
      file_mask = file_mask.replace('.json', '.png')
      w, h, labels, vertexlist = get_poly(file_json)
      img2 = Image.new("RGB", (w, h))
      img12 = ImageDraw.Draw(img2)
      for i in range(len(vertexlist)):
        #'data/mask/354/frame0011_gtFine_polygons.json' is having single pair of coordinates--
        #--for one 'vegetation' type label and also empty lists: Polygon can't be drawn in such case--
        #--ImageDraw.Draw.polygon() Method needs Sequence of 2-tuples like [(x, y), (x, y), …]
        #--Therefore Ignoring that and such type of Polygons
        if len(vertexlist[i])==1 or len(vertexlist[i])==0:       
          continue
        img12.polygon(vertexlist[i], fill = label_clr[labels[i]])
      img2=np.array(img2)
      dir_mask = ('/').join(file_mask.split(sep='/')[0:3])
      #img4 = Image.fromarray(img2, 'RGB')
      if not os.path.exists(dir_mask):
          os.makedirs(dir_mask)
      img4 = Image.fromarray(img2[:,:,0])
      img4.save(file_mask)
      masks_all.append(file_mask)
    data_df['mask'] = masks_all
      # img5 = mpimg.imread(file_mask)      ## Mask Image
      # #imgplot = plt.imshow(img5[:,:,0])  ## Mask Image
      # img222 = mpimg.imread(file_img)     ## Original Image
      # #imgplot22 = plt.imshow(img222)     ## Original Image
    return data_df

from tqdm import tqdm
data_df = compute_masks(data_df)
data_df.head()

import urllib.request
def grader_3():
    url = "https://i.imgur.com/4XSUlHk.png"
    url_response = urllib.request.urlopen(url)
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, -1)
    my_img = cv2.imread('data/output/201/frame0029_gtFine_polygons.png')    
    plt.imshow(my_img)
    print((my_img[:,:,0]==img).all())
    print(np.unique(img))
    print(np.unique(my_img[:,:,:]))
    data_df.to_csv('preprocessed_data.csv', index=False)
grader_3()