<!DOCTYPE html>
<html>
<head>
</head>
<body>
  <h1>Lung Cancer Detection Model Training</h1>
  <p>This project focuses on training a deep learning model for lung cancer detection using medical images. The pipeline involves data preprocessing, model creation using a convolutional neural network (CNN), and training the model on the dataset. Below is a detailed overview of the key steps involved:</p>

  <h2>Dataset Preparation and Preprocessing</h2>
  <h3>Dataset Description:</h3>
  <ul>
    <li>The dataset consists of medical images of lung scans, categorized into three classes: Bengin_cases, Malignant_cases, and Normal_cases.</li>
    <li>Each image in the dataset represents a lung scan with varying sizes and resolutions.</li>
  </ul>

  <h3>Data Loading and Preprocessing:</h3>
  <ul>
    <li>Images are loaded using OpenCV and converted to grayscale.</li>
    <li>Preprocessing steps include resizing the images to a fixed size (e.g., 256x256), normalizing pixel values to be between 0 and 1, and augmenting the dataset using techniques like rotation, flipping, and scaling to increase dataset diversity.</li>
  </ul>

  <h3>Labeling and Class Balancing:</h3>
  <ul>
    <li>Labels for each image are assigned based on the classification into Bengin, Malignant, or Normal cases.</li>
    <li>Class balancing techniques like SMOTE are applied to address class imbalance issues in the dataset.</li>
  </ul>

  <h2>Model Architecture</h2>
  <h3>Transfer Learning with VGG16:</h3>
  <ul>
    <li>Transfer learning is utilized with the VGG16 (Visual Geometry Group 16) model pretrained on the ImageNet dataset for feature extraction. This helps in leveraging the learned features from a large dataset for better performance on the lung cancer detection task.</li>
  </ul>

  <h3>Additional Layers:</h3>
  <ul>
    <li>Additional layers are added on top of the VGG16 base to adapt it to the lung cancer detection task. These layers include a Flatten layer, a Dense layer with 256 units and ReLU activation, a Dropout layer with a dropout rate of 0.5 to prevent overfitting, and a Dense layer with 3 units and softmax activation for class prediction (benign, malignant, normal).</li>
  </ul>
<h2>Pretrained Model</h2>

<p>If you prefer to use a pretrained model for lung cancer detection, you can follow these steps:</p>

<ol>
  <li><strong>Download the Pretrained Model:</strong> Obtain the pretrained model file (<code>lung_cancer_model.h5</code>) from the source provided.</li>
  
  <li><strong>Load the Pretrained Model:</strong> Use a deep learning framework like Keras or TensorFlow to load the pretrained model into your environment.</li>
  
  <li><strong>Preprocess Input Images:</strong> Define a function to preprocess the input images before feeding them into the pretrained model. This may include resizing, normalization, and other preprocessing steps.</li>
  
  <li><strong>Make Predictions:</strong> Use the loaded pretrained model to make predictions on new images. The model will output predictions indicating whether the input images contain benign cases, malignant cases, or normal cases of lung cancer.</li>
</ol>
  <p>Accuracy of the pre-trained Model:</p>
  <img src = "https://github.com/anandakumarak/LUNG-CANCER-DETECTION-USING-CNN/blob/main/Model%20Accuracy.png" alt="Accuracy" width="700">

<p>You can customize the above steps based on your specific requirements and environment.</p>

  <h2>Model Training</h2>
  <h3>Running the Code:</h3>
  <ol>
    <li>Organize Dataset: Ensure your dataset is organized in the specified folder structure.</li>
    <li>Update File Paths and Parameters: Update the file paths and parameters in the code as needed. This includes the paths to your dataset, the model file, and any other relevant parameters such as image size, batch size, and number of epochs.</li>
    <li>Preprocess Images: Run the code to preprocess the images, including resizing, normalization, and augmentation.</li>
    <li>Create and Train Model: Create the model using transfer learning with VGG16 and additional layers. Compile the model with appropriate loss function and optimizer. Train the model using the augmented dataset.</li>
    <li>Evaluate Model Performance: Evaluate the model's performance on the validation set. Monitor metrics such as loss and accuracy to assess the model's performance.</li>
    <li>Save Model: If satisfied with the model's performance, save the trained model for future use.</li>
    <li>Run the Code: Run the updated code to preprocess the images, create and train the model, and evaluate its performance. Make sure to monitor the training progress and adjust parameters as needed for better performance.</li>
  </ol>
  <h2>Sample Output</h2>
  <p>Below is an example of the model's prediction:</p>
  <img src = "https://github.com/anandakumarak/LUNG-CANCER-DETECTION-USING-CNN/blob/main/Model%20Output.png" alt="Sample Output Image" width="700">
  <p>The green color box in the above output image is not a prediction by the model; it is present in the test images.</p>

  <h2>Future Work</h2>
  <p>In future work, I plan to develop a Mask R-CNN (Mask Region-based Convolutional Neural Network) model to detect lung cancer by marking the regions where cancer cells are present.</p>
  
</body>
</html>

