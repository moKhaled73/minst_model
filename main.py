from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('minst_model.h5')

app = FastAPI()

@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()
        # print(image_data)
        image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
        
        # Resize and preprocess the image
        image = image.resize((28, 28))
        image_array = np.array(image, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0

        # Make prediction
        prediction = model.predict(image_array)
        predicted_label = np.argmax(prediction)

        return {"predicted_label": int(predicted_label)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




@app.get("/")
def root():
    return {'hello': 'world'}