# BIG-DATA-CW2
## Classifying digits from (0-9) from a 32x32RGB image using a loaded CNN model.
CW2 with Abed and Jack
## Introduction
This project focuses on building a complete machine learning (ML) pipeline for classifying digits (0–9) using a Convolutional Neural Network (CNN). The dataset used is the Street View House Numbers (SVHN), which consists of 32x32 RGB images of digits cropped from natural scene images. The objective is to deploy a reliable digit classification model with a strong emphasis on pipeline development, evaluation, and deployment of CNNs.

## Business Objectives
The goal is to accurately classify individual digits from real-world image data. High classification accuracy is essential, as such a system could be applied in real-time digit recognition scenarios like address or license plate recognition. The target is to achieve over 90% accuracy on unseen test data.

## ML Pipeline

### 1. Data Collection
The project begins by importing two foundational Python modules: os and urllib.request. The os module provides functionality to create and manage directories in a way that works across different operating systems, which is important for organizing and accessing the dataset. The urllib.request module enables the script to make HTTP requests, which is essential for downloading the SVHN dataset files directly from Stanford's server. This approach ensures the entire pipeline is reproducible and does not require manual data setup.

To organize the downloaded data, the script creates a new directory called data using the os.makedirs() function. By passing the parameter exist_ok=True, the code avoids throwing an error if the folder already exists. This design makes the script idempotent, allowing it to be safely re-run without duplicating effort or interrupting the workflow due to existing directory conflicts.

The script defines a dictionary named urls that contains the download links for three subsets of the SVHN dataset: training, testing, and extra data. Each key in the dictionary corresponds to a specific subset and maps to the appropriate URL. This method allows for easy iteration during the download process and keeps the dataset sources well-organized and centralized within the code, making updates or changes simple and scalable.

To automate the retrieval of the SVHN data, a function called download_svhn() is defined. This function loops through each item in the urls dictionary and checks whether the corresponding dataset file already exists in the data directory. If a file is missing, it uses urllib.request.urlretrieve() to download it and saves it with a name like train_32x32.mat. Progress messages are printed during each download to provide user feedback. This function streamlines the initial setup and ensures that all required datasets are locally available before further processing.

Several critical libraries are imported next to support data handling, visualization, and machine learning. The scipy.io module is used to read .mat files, which is the format in which the SVHN dataset is provided. numpy is employed for numerical operations and array manipulations. The train_test_split function from sklearn is utilized to divide the dataset into training and validation subsets. For visualizations, matplotlib.pyplot allows the display of images and plots. Lastly, to_categorical from tensorflow.keras.utils is used to convert numeric digit labels into one-hot encoded vectors, a format required by many neural network architectures for classification tasks.

The script then loads the training and testing datasets using scipy.io.loadmat(), which reads MATLAB-formatted .mat files into Python dictionaries. Each dictionary contains an image array and a corresponding label array. For example, train_data['X'] contains the training images, while train_data['y'] holds the labels. These datasets are now ready to be preprocessed and reshaped for use in a convolutional neural network.

### 2. EDA
Before model training begins, the script prints out the shapes of the training, validation, and test datasets. This serves as a sanity check to ensure that the preprocessing pipeline has correctly formatted the data. It verifies that the inputs have four dimensions (samples, height, width, channels) and that the labels align with the number of samples.

The code visualizes five random images from the training dataset using matplotlib. Each image is displayed with its corresponding class label as the title. This step is part of exploratory data analysis (EDA), providing a visual confirmation that the data looks as expected and has been loaded and processed correctly. It also helps identify any anomalies or incorrect labels early in the workflow.

### 3. Model Building
The image data loaded from the .mat files initially has a shape of (32, 32, 3, N), where N is the number of images. However, most deep learning frameworks, including Keras, expect image data to follow the format (N, 32, 32, 3). To accommodate this, the script transposes the image arrays using np.transpose() and adjusts the axis ordering. Additionally, label arrays are flattened from shape (N, 1) to (N,) using flatten(), which simplifies further processing. Notably, there is a small error in this section where y_test is mistakenly assigned from train_data['y'] instead of test_data['y'].

The SVHN dataset represents the digit '0' using the label '10', which can be confusing during model training. To correct this, the script replaces every instance of label 10 with 0 in both the training and test label arrays. This step ensures consistency with standard digit labels and avoids logical mismatches during training and evaluation.

To enhance model performance and ensure consistent input scaling, the pixel values of the image data are normalized. Since the original RGB values range from 0 to 255, they are converted to a floating-point format (float32) and divided by 255.0. This transformation scales the pixel intensities to a range of 0 to 1, which is known to improve convergence speed and training stability in neural networks.

The script also includes a procedure to load the optional extra dataset provided by SVHN, which can be used to further train or augment the model. Like the main training set, the labels in this dataset are flattened and corrected by converting 10s to 0s. Due to its large size, the extra data is processed in batches of 10,000 samples to prevent memory overflow. Each batch is reshaped and normalized in the same way as the primary datasets. This approach allows the extra data to be used effectively without compromising system performance.

The script uses train_test_split() from sklearn.model_selection to divide the original training dataset into two subsets: training and validation. This is a crucial step in machine learning pipelines, as it allows for evaluation of model generalization on unseen data. Specifically, 80% of the data is used for training, while 20% is set aside for validation. The stratify=y_train parameter ensures that the class distribution remains consistent across both subsets, and the random_state=42 guarantees reproducibility of the split.

Before feeding the data into the CNN, the class labels for both training and validation subsets are converted into one-hot encoded vectors using to_categorical(). This transformation converts integer labels (e.g., 2) into binary class matrices (e.g., [0, 0, 1, ..., 0]), which is a standard format required by many classification models in Keras. This allows the neural network to compute categorical cross-entropy loss correctly.

To reduce training time during development, the dataset is manually downsampled. Only the first 10,000 samples from the training split and the first 2,000 from the validation split are retained. This is a common practice during prototyping to enable faster iterations before scaling to the full dataset. While not ideal for final accuracy, it’s useful for early debugging and architecture tuning.

Before model training begins, the script prints out the shapes of the training, validation, and test datasets. This serves as a sanity check to ensure that the preprocessing pipeline has correctly formatted the data. It verifies that the inputs have four dimensions (samples, height, width, channels) and that the labels align with the number of samples.

The code visualizes five random images from the training dataset using matplotlib. Each image is displayed with its corresponding class label as the title. This step is part of exploratory data analysis (EDA), providing a visual confirmation that the data looks as expected and has been loaded and processed correctly. It also helps identify any anomalies or incorrect labels early in the workflow.

In this section, the script redundantly re-applies one-hot encoding to the training and validation labels using to_categorical(). This appears to duplicate the step from cell 12. While not harmful, this redundancy could be cleaned up for clarity and efficiency.

A Convolutional Neural Network (CNN) is constructed using Keras' Sequential API. The architecture includes two convolutional layers with ReLU activation, each followed by a max-pooling layer to reduce spatial dimensions. The flattened output is passed through a dense layer with 128 units and a dropout layer to prevent overfitting. Finally, a softmax output layer maps the feature representations to 10 digit classes. This structure balances depth and simplicity, suitable for the 32x32 pixel images.

This cell repeats the dataset downsampling from cell 13, again slicing the first 10,000 and 2,000 samples for training and validation respectively. This repetition could be a remnant of iterative testing and should ideally be removed or consolidated.

### 4. Model Evaluation
The CNN model is compiled using the Adam optimizer and categorical cross-entropy loss, which is appropriate for multi-class classification. The training is initiated with a batch size of 64 and runs for 5 epochs. Both the training and validation sets are used, enabling the model to report progress on unseen data after each epoch. The resulting history object stores accuracy and loss metrics for later visualization

Finally, the script plots the training and validation accuracy and loss across the 5 training epochs. This provides visual insight into the model's learning dynamics. If the training and validation metrics diverge significantly, it may indicate overfitting. Conversely, steady improvements in both suggest that the model is learning effectively. These plots are crucial for diagnosing model behavior and guiding further tuning.

This section imports libraries that are essential for evaluating model performance. classification_report and confusion_matrix from sklearn.metrics provide textual and matrix-based summaries of how well the model performs across all classes. seaborn is imported for high-level visualizations and is specifically used for creating an annotated heatmap of the confusion matrix. matplotlib.pyplot is again used to render the plots. These tools help convert raw predictions into understandable performance metrics.

To compute classification metrics, the script first converts the one-hot encoded validation labels (y_val_cat) back to their original integer form using np.argmax(axis=1). This is applied to both the ground-truth labels and predictions. However, a small bug exists: both y_val_true and y_val_pred are derived from the same y_val_cat, which results in identical vectors. The model's actual predictions should be obtained using model.predict() on X_val_split, followed by argmax to extract the predicted class indices. As it stands, this code mistakenly compares the validation set against itself, artificially inflating performance metrics.

A textual classification report is printed using classification_report() from sklearn. It provides metrics such as precision, recall, and F1-score for each digit class. These metrics offer a granular view of model accuracy per class, revealing any biases or performance weaknesses. However, due to the aforementioned bug, the report in its current form reflects perfect accuracy and must be corrected to reflect true model performance.

A confusion matrix is created and visualized using seaborn.heatmap(). This matrix shows the number of correct and incorrect predictions for each class in a grid format, with the rows representing true labels and columns representing predicted labels. Annotated values within the matrix cells allow for quick identification of common misclassifications. However, because the prediction vector is flawed in this implementation, the confusion matrix is effectively a diagonal matrix, misleadingly indicating perfect classification. Once corrected, this matrix will be a valuable diagnostic tool for understanding model behavior.
### 5. Prediction

## Jupyter Notebook Structure

## Future Work

## Libraries and Modules

## Unfixed Bugs

## Acknowledgements and References

## Conclusions
