from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image
import numpy as np
import io

model = load_model('models/Mobilenetcnn.h5')

# Transformations for the input image
def transform_image(image):
    img = load_img(image, target_size=(128, 128))  # Resize the image to 128x128 pixels
    img = img_to_array(img)  # Convert the image to a NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255.0  # Normalize pixel values
    return img

# Class names for predictions
num_classes = ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']

def predict_image(img):
    # Open the image from bytes
    img_pil = Image.open(io.BytesIO(img))
    
    # Resize the image to the required input size of the model (e.g., 128x128)
    img_resized = img_pil.resize((128, 128))
    
    # Convert the image to array
    img_array = img_to_array(img_resized)
    
    # Expand dimensions to match the input shape of the model
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Preprocess the input image array
    img_array_preprocessed = preprocess_input(img_array_expanded)
    
    # Make prediction
    predictions = model.predict(img_array_preprocessed)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions[0])
    
    # List of class names (adjust as per your dataset)
    num_classes = ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']

    # Return the predicted class name
    return num_classes[predicted_class_index]
