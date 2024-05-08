import numpy as np
import cv2
from keras.models import load_model

# Load the saved model
loaded_model = load_model('D:\\PROJECTS\\LUNG_CANCER\\code_lung\\lung_cancer_model.h5')

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
input_image_path = 'D:\\Projects\\Lung_Cancer\\dataset\\Test cases\\000058_07_01_170.png'

# Make prediction
prediction, accuracy = predict_cancer(input_image_path)

# Read the image
image = cv2.imread(input_image_path)
print('Prediction :' ,prediction)
print('Accuracy :',accuracy)
# Display the image with predicted label and accuracy
cv2.putText(image, f"Prediction: {prediction}, Accuracy: {accuracy:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
