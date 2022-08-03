from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from streamlit_chat import message as st_mesage
import  streamlit as st
from PIL import Image
import os

model3 = core.Model.load('model3_weights.pth', ['idly','dosa'])
st.title('Hi Santhosh!!!')
st_mesage('Upload your Image....')
def prediction(x):
  model3 = core.Model.load('model3_weights.pth', ['idly','dosa'])
  image = utils.read_image(x)
  predictions = model3.predict(image)
  labels, boxes, scores = predictions
  thresh=0.6
  filtered_indices=np.where(scores>thresh)
  filtered_scores=scores[filtered_indices]
  filtered_boxes=boxes[filtered_indices]
  num_list = filtered_indices[0].tolist()
  filtered_labels = [labels[i] for i in num_list]
  return {'image_result':show_labeled_image(image, filtered_boxes, filtered_labels),'Items_Presents':list(set(filtered_labels))}





uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file!=None:
    with open(os.path.join("tempDir", 'testing.jpg'), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Saved File")
    file=os.listdir('tempDir')
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width='auto',width=5)
    st_mesage('File Uploaded',is_user=True)
    label = prediction('tempDir/testing.jpg')
    if len(label['Items_Presents'])==2:
        st_mesage('Do You want '+label['Items_Presents'][0]+' Or '+label['Items_Presents'][1])
    elif len(label['Items_Presents'])==1:
        st_mesage('Do You want '+label['Items_Presents'][0])
    elif label==None:
        pass


    #st.write(label['Items_Presents'])
    #st.image(Image.open(label['image_result']))



