import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import cv2
import tensorflow as tf
from metrics import f1
import import_ipynb
# from ipynb.fs.full.MIR import train
# import MIR

bodyparts = {
0 : 'Abdomen' ,
1 :'Ankle' ,
2 :'Cervical Spine',
3 : 'Chest' ,
4 :'Clavicles' ,
5 :'Elbow' ,
6 :'Feet' ,
7 : 'Finger' ,
8 : 'Forearm' ,
9 : 'Hand' ,
10 : 'Hip' ,
11 : 'Knee' ,
12 : 'Lower Leg' ,
13 : 'Lumbar Spine' ,
14 : 'Others' ,
15 :'Pelvis',
16 :'Shoulder' ,
17 :'Sinus' ,
18 : 'Skull' ,
19 : 'Thigh' ,
20 :'Thoracic Spine',
21: 'Wrist',
}


def predict(image):
  baseUrl = "testDir/Prediction/"
  img = cv2.imread(baseUrl+image)
  img = cv2.resize(img,(224,224))
  img = np.reshape(img,[224, 224, 3])
  # images_list = []
  images_list = []
  images_list.append(np.array(img))
  x = np.asarray(images_list)

  loaded_model = tf.keras.models.load_model('mobileNet-78.h5', custom_objects={"f1": f1})
  predict = loaded_model.predict(x)
  # classes_x=np.argmax(predict,axis=1)
  return predict


st.set_page_config(
   page_title="Medical Imaging Router",
   #page_icon=logo,
   #layout="wide",
   #initial_sidebar_state="expanded",
)

st.title('Medical Imaging Router')

testImage1 = Image.open("chest-scan.png")
testImage2 = Image.open("chest-scan-mod.png")

img_arrays = [testImage1, testImage2]
placeholder = st.empty()
for img_array in img_arrays:
  placeholder.image(img_array)
  time.sleep(5)


# st.image(testImage1, caption="Chest scan before processiong")
# st.image(testImage2, caption="Chest scan after processiong")

uploaded_file = st.file_uploader("Choose a file(s)", accept_multiple_files=True)
# st.write(uploaded_file)

if uploaded_file is None:
  st.write('No file(s) selected')
else:
  predictBtn = st.button("Predict", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False)
  if predictBtn:
    for i in uploaded_file:
      result = predict(i.name)
      classes = np.array(['Abdomen', 'Ankle', 'Cervical Spine', 'Chest', 'Clavicles', 'Elbow', 'Feet', 'Finger', 'Forearm', 'Hand', 'Hip', 'Knee',
                      'Lower Leg', 'Lumbar Spine', 'Others', 'Pelvis', 'Shoulder', 'Sinus', 'Skull', 'Thigh', 'Thoracic Spine', 'Wrist'])
      # result = np.append(result, classes, axis=1)
      # result = np.insert(result, 1, classes, axis=1)
      result_df = pd.DataFrame(result)
      # st.write(result)
      # st.write(type(result_df))
      result_df = result_df.append(pd.Series(classes), ignore_index = True)
      result_df = result_df.apply(np.roll, shift=1)
      st.write(result_df)

      label = result.argmax()
      st.write("Predicted class: ", bodyparts[label])



