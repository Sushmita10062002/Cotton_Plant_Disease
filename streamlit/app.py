
import numpy as np
import os
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image

path = os.path.dirname(__file__)
model_file = path+"/cotton_plant_disease_model.h5"
model = load_model(model_file)

def preprocess_image(img):
  img_arr = image.img_to_array(img)
  img_arr = img_arr/255
  img_arr = np.expand_dims(img_arr, axis=0)
  return img_arr
def prediction(img):
  size = (150,150)
  img = img.resize(size)
  processed_img = preprocess_image(img)
  result = model.predict(processed_img).round(3)
  preds = np.argmax(result)
  if preds==0:
    preds="The leaf is diseased cotton leaf"
    return preds
  elif preds==1:
    preds="The leaf is diseased cotton plant"
    return preds
  elif preds==2:
    preds="The leaf is fresh cotton leaf"
    return preds
  else:
    preds="The leaf is fresh cotton plant"
    return preds

  
def main():
    st.title("Cotton Plant Disease Prediction")
    html_temp = """
    <div style="background-color:#069A8E; padding:5px; margin-bottom:2rem">
    <h3 style="color:white; text-align:center;">Prediction of disease in cotton plant via images using CNNS</h3>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    upload_file = st.file_uploader("Upload Cotton plant image", type=['png', 'jpg', 'jpeg'])
    if st.button("Predict"):
        img = Image.open(upload_file)
        with st.expander('Image', expanded = True):
            st.image(img, use_column_width=True)
        result = prediction(img)
        st.title(result)

if __name__ == '__main__':
    main()
