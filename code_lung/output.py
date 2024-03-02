import numpy as np
import cv2
from keras.models import load_model

# Load the saved model
loaded_model = load_model('lung_cancer_model.h5')

# Function to preprocess input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))  # Assuming VGG16 input shape
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    return img

# Function to make predictions
def predict_cancer(image_path):
    # Preprocess the input image
    img = preprocess_image(image_path)

    # Make predictions
    predictions = loaded_model.predict(img)
    class_labels = ['Bengin_cases', 'Malignant_cases', 'Normal_cases']
    predicted_label = class_labels[np.argmax(predictions)]
    accuracy = np.max(predictions)

    # Return the predicted label and accuracy
    return predicted_label, accuracy

# Provide the path to your input image
input_image_path = 'D:\\PROJECTS\\LUNG_CANCER\\dataset\\The_IQ_OTHNCCD_lung_cancer_dataset\\The_IQ_OTHNCCD_lung_cancer_dataset\\Normal_cases\\Normal case (105).jpg'

# Make prediction
prediction, accuracy = predict_cancer(input_image_path)

# Load and display the input image
img = cv2.imread(input_image_path)
cv2.imshow('Input Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the predicted label and accuracy
print("Prediction:", prediction)
print("Accuracy:", accuracy)
