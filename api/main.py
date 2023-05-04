from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from scipy.spatial.distance import cosine

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

width = 256
height = 256
MODEL = tf.keras.models.load_model("D:/PotatoDisease/models/1")
CLASS_NAMES = ["Early Blight","Late Blight","Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, Im here"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
    

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
     
    ##New Code
    #image_array = np.array(Image.open(image))
    resize_image = cv2.resize(image,(width, height))
    img_batch = np.expand_dims(resize_image,0)
    
    ##End  
    #img_batch = np.expand_dims(image,0)#Old Code
    
    try:
        
    
        predications = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predications[0])]
        confidence = np.max(predications[0])
        
        return{
          'class':predicted_class,
          'confidence':float(confidence),
          
        }
        
    except ValueError as e:
        print("Image Shape is Not in formated",str(e) )   



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8080)