from fastapi import FastAPI, UploadFile, File, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="ResNet18 ONNX Inference API")

# Load the ONNX model at startup
ort_session = ort.InferenceSession("resnet18_deployed.onnx", providers=['CPUExecutionProvider'])

def preprocess_image(image_bytes):
    """
    ResNet expects images to be resized, cropped, and normalized in a very specific way.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((256, 256)) # Standard ResNet resize
    
    # Center crop to 224x224
    width, height = image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # Convert to numpy array and scale to [0, 1]
    img_data = np.array(image).astype('float32') / 255.0
    
    # Normalize with ImageNet mean and std deviation
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    img_data = (img_data - mean) / std
    
    # Transpose to (Channels, Height, Width) -> (3, 224, 224)
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Add batch dimension -> (1, 3, 224, 224)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data.astype('float32')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        # Read the image bytes
        image_bytes = await file.read()
        
        # Preprocess the image to match ResNet's expectations
        input_tensor = preprocess_image(image_bytes)
        
        # Run inference using ONNX Runtime
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Get the predicted class index
        prediction = ort_outs[0]
        predicted_class_idx = np.argmax(prediction).item()
        
        return {"predicted_class_id": predicted_class_idx, "message": "Success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "ONNX ResNet18 API is up and running!"}