Data Collection:

This data is from kaggle. https://www.kaggle.com/datasets/ritvik1909/document-classification-dataset

The original specifies :
This is a small subset of rvl cdip dataset created to try out Document Classification Algorithms
It contains 3 out of 16 classes present in the rvl cdip dataset:

Data Processing:

The code loads image data from a specified directory structure and creates a DataFrame (data) containing information about each image, including its path, filename, and kind (label).
The images are then split into training and testing sets
Image data is preprocessed using TensorFlow functions (process_img) to read, decode, resize, and normalize the images.


Model Choice:

The chosen model architecture is MobileNetV2, a pre-trained convolutional neural network (CNN) available through TensorFlow Hub.
Transfer learning is used, where the pre-trained MobileNetV2 is employed as a feature extractor, and a dense layer is added as output layer for classification.
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss. Custom metrics (f1_m, precision_m, recall_m) are included for monitoring training.


The choice of MobileNetV2 for this image classification task is motivated by its efficiency and adaptability. Leveraging transfer learning from a pre-trained MobileNetV2 model allows the network to capture hierarchical features from images, making it particularly suitable for classifying diverse images such as resumes. Moreover, MobileNetV2's resource efficiency, speed of inference, and suitability for deployment align well with the practical considerations of classifying document images.


Training:

Training data is batched and preprocessed using the d_batch function, which applies shuffling for training data.
Early stopping and TensorBoard callbacks are utilized during training to monitor loss and stop early if overfitting is detected.
The model is trained for a maximum of 100 epochs.


Evaluation:

The model is evaluated on the test set, and metrics such as loss, accuracy, F1 score, precision, and recall are computed.
The confusion matrix is generated and visualized using Seaborn to analyze the model's performance on different classes.
The trained model is saved, including a timestamp in the saved model's filename.


Results:

The training results indicate that the model achieves high accuracy and a reasonable F1 score on the training set.
The test results show decent performance, with a good F1 score, accuracy, and precision. However, recall on the training set seems unusually high. (Very much due to small size of data)
TensorBoard is used to visualize the training process and monitor metrics over time

training loss: 0.1158 - acc: 0.9773 - f1_m: 0.8219 - precision_m: 0.6870 - recall_m: 1.0227

test loss: 0.4662 - accuracy: 0.7879 - f1 : 0.9038 - precision : 0.8387 - recall : 1.0000