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

---
## 4. Model Evaluation
The two convolutional nueral networks (CNNs) were trained nd evaluated on the SVHN (Street View House Numbers) Dataset.

**Model 1 - Baseline CNN**: introduces a simple two layer archetecture with dropout for regularization.

**Model 2 - Optimized CNN** : Introduces a deeper model using 3 convolutional layers and BatchNormalization for improved optimization.

### performance metrics

| Metric            | Baseline CNN | Optimized CNN |
|-------------------|--------------|----------------|
| Validation Accuracy | 86%         | 89%             |
| Macro F1-score    | 0.86         | 0.88            |
| Weighted F1-score | 0.86         | 0.89            |


### Observations
The optimised CNN outperformed the baseline model across accuracy, recall and F1-score. Confustion matrices were created to show reduced missclasification on similar digits e.g.(3 vs 5, or 4 vs 9) for the optimized model, allowing visual predictions to highlight the models strong confidence on clean digits and uncertanty on blurred or missaligned samples. Thus both models demonstrating solid genralisation, but the optimized version was clearly more robust and stable compared to the baseline model.


The CNN models are compiled using the Adam optimizer and categorical cross-entropy loss, which is appropriate for multi-class classification. The training for CNN(1) is initiated with a batch size of 64 and runs for 10 epochs. CNN(2) is then initiated with a batch size of 64 and runs for 15 epochs, Both the training and validation sets are used, enabling the models to report progress on unseen data after each epoch. The resulting history object stores accuracy and loss metrics for later visualization.

#### This shows the CNN(1) Baseline model.

            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

            model_base = Sequential([
                Conv2D(32,(3, 3), activation = 'relu', input_shape = (32, 32, 3)),
                MaxPooling2D(),

                Conv2D(64,(3, 3), activation = 'relu'),
                MaxPooling2D(),

                Flatten(),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(10, activation='softmax')

            ])


            model_base.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

            history_base = model_base.fit(
                X_train_split, y_train_cat,
                validation_data = (X_val_split, y_val_cat),
                epochs = 10,
                batch_size = 64
                )

#### This shows the CNN(2) Optimized model.
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

                model_opt = Sequential([
                    Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
                    BatchNormalization(),
                    MaxPooling2D(),


                    Conv2D(64, (3, 3), activation = 'relu'),
                    BatchNormalization(),
                    MaxPooling2D(),


                    Conv2D(128, (3, 3), activation = 'relu'),
                    BatchNormalization(),
                    MaxPooling2D(),

                    Flatten(),
                    Dropout(0.4),
                    Dense(128, activation = 'relu'),
                    Dropout(0.3),
                    Dense(10, activation = 'softmax')
                ])



                model_opt.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

                history_opt = model_opt.fit(
                    X_train_split, y_train_cat,
                    validation_data = (X_val_split, y_val_cat),
                    epochs = 15,
                    batch_size = 64,
                    callbacks = [early_stop, checkpoint]
                )


Finally, the script plots the training and validation accuracy and compare the two models the baseline and the optimized model, this provides visual insight into the model's learning dynamics. If the training and validation metrics diverge significantly, it may indicate overfitting. Conversely, steady improvements in both suggest that the models are learning effectively but the optimized model overseaing the baseline model because of its optimization. These plots are crucial for diagnosing model behavior and guiding further tuning. the differnces between the acuracy and loss comparisson between the baseline and optimized model can be seen within the graph generated by the scrippt above.

A confusion matrix is then generated and visualized using seaborn.heatmap(). This matrix shows the number of correct and incorrect predictions for each class in a grid format, with the rows representing true labels and columns representing predicted labels. Annotated values within the matrix cells allow for quick identification of common misclassifications. The matrixs will be generate for both models allowing a visual comparison to be availble to be able to determine the differnces in the accuracy in the models and how optimization has helped the accuracy improved.

This section imports libraries that are essential for evaluating model performance. classification_report and confusion_matrix from sklearn.metrics provide textual and matrix-based summaries of how well the models perform across all classes. seaborn is imported for high-level visualizations and is specifically used for creating an annotated heatmap of the confusion matrix. matplotlib.pyplot is again used to render the plots. These tools help convert raw predictions into understandable performance metrics.

                from sklearn.metrics import confusion_matrix, classification_report
                import pandas as pd
                import seaborn as sns
                import matplotlib.pyplot as plt
                import numpy as np

                # Predicting and converting from one hot to label
                y_val_true = np.argmax(y_val_cat, axis = 1)
                y_val_pred_base = np.argmax(model_base.predict(X_val_split), axis = 1)


                # Confusion Matrix
                cm_base = confusion_matrix(y_val_true, y_val_pred_base)

                plt.figure(figsize = (8, 6))
                sns.heatmap(cm_base, annot = True, fmt = 'd', cmap = 'Blues', xticklabels= range(10), yticklabels= range(10))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix - Baseline CNN')
                plt.show()

                # Classifcation report

                report_base = classification_report(y_val_true, y_val_pred_base, output_dict = True)
                report_df_base = pd.DataFrame(report_base).transpose().round(2)
                report_df_base

A classification repoprt of each confusion matrix is generated. using the code below generated the classifcation report for the CNN(1) model providing metrics such as precision, recall and F1-score for each digit class generated.These metrics offer a granular view of model accuracy per class, revealing any biases or performance weaknesses. However, due to the aforementioned bug, the report in its current form reflects perfect accuracy and must be corrected to reflect true model performance.

The same process is then applied for the optimized model allowing for a comparison between the two models and clear representation as to what the accuracy differnce is between the two models and taking into consideration the factors of optimization. 

                y_val_pred_opt = np.argmax(model_opt.predict(X_val_split), axis = 1)

                # Confusion Matrix
                cm_opt = confusion_matrix(y_val_true, y_val_pred_opt)

                plt.figure(figsize = (8, 6))
                sns.heatmap(cm_opt, annot = True, fmt = 'd', cmap = 'Blues', xticklabels= range(10), yticklabels= range(10))
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Confusion Matrix - Optimised CNN')
                plt.show()



                # Classifcation report

                report_opt = classification_report(y_val_true, y_val_pred_opt, output_dict = True)
                report_df_opt = pd.DataFrame(report_opt).transpose().round(2)
                report_df_opt
---
## 5. Prediction
After training, both CNN models were evaluated on previously unseen data, The optimized moel was also used to generate predictions on a subset allowing for idividual test images to be generated. 

### Visual Results
A selection of 5-10 digits were dislayed with generated predicted labels, confidence scores and true labels allowing for strong confidence for clean centered digits. Missclassifications typically involved similar digits such as 4 and 5 being missclassified as 9 and 3. 

                import matplotlib.pyplot as plt
                from tensorflow.keras.models import load_model

                # Loading the trained model
                model = load_model('best_model.keras')

                # Running predictions based on the test set
                y_pred_probs = model.predict(X_test)
                y_pred = np.argmax(y_pred_probs, axis = 1)

                # Plotting predictions
                plt.figure(figsize = (12, 4), dpi = 120)

                for i in range(5):
                    plt.subplot(1, 5, i + 1)
                    plt.imshow(X_test[i], cmap = 'grey')
                    plt.title(f"Pred: {y_pred[i]}\nConf: {y_pred_probs[i][y_pred[i]]:.2f}", fontsize = 10)
                    plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}\nConf: {y_pred_probs[i][y_pred[i]]:.2f}", fontsize=10)
                    plt.axis("off")

                plt.suptitle("SVHN Model Predictions on Test Images", fontsize = 14)
                plt.tight_layout()
                plt.show()

Along side the visual result an error analysis was carried out to show incorrect predictions were possible, errors of misclassification occured when discrepecies such as low-resolution, low contrast, blurry images and digits that were obscured or slightly overlapping, caused for an error to be presented. Thus generating an image that was hard to decipher. 

            # Find a wrong pediction
            wrong_idx = np.where(y_val_pred_opt != y_val_true)[0]

            # Visulise first incorrect one to show that the model can get things wrong
            i = wrong_idx[0]
            plt.imshow(X_val_split[i])
            plt.title(f"True: {y_val_true[i]}, pred: {y_val_pred_opt[i]}")
            plt.axis("off")
            plt.show()
---
### Jupyter Notebook Structure

 This project is implemented in a single jupyter notebook file and organizedinto the following steps:
 1. Data Collection Downloaded from Urls on the SVHN Website
 2. preproces the Svhn Dataset
 3. Training of the CNN Models
 4. Evaluation of the Trained CNN Model
 5. Predictions
 6. Conclusion 


---
### Future Work

1. Aditional archtechtures could be implemnted such as RedNet or MobileNet for etter feature extration.
2. use data augmentation to improve model robustness on rotated or occluded digits
3. perform certain hyper perameters tuning grid searches or byesian opimization.
4. implement real-time digit ecognition using a webcam input and opencv
5. depploy the model via a web app using things such as streamlit or flask to present it.
---
### Libraries and Modules

Key libaries used in this project were:
1. TensorFlow /Keras - Deep learning for building the CNN/
2. NumPy - Effiecent array opperations/
3. Pandas - data handeling for classification reports/
4. MatPlotlib / seaborn - visulization(EDA, Confusion matrices, training curves)
5. Scikit-learn - Evaluation metrics for classification reports and confusion matrices.
6. Scipy.io - for loading .mat SVHN files.

---
### Unfixed Bugs

1. Occasonally large .npz data chunks can cause memory warnings due tolimited RAM systems for full operation to take place.
2. classification reports cansometimes crash the kernal if run on the entire test set without sampeling.
3. some minor prediction errors occur on extreamly noisy digits this is common on the SVHN data set due to real world image noise.
4. the kernal can someties class when allowing batch size to be above certain values 10,000+ this can cause an overload on the kernal causing it to be killed.

---
### Acknowledgements and References
1. SVHN Dataset: http://ufldl.stanford.edu/housenumbers/


### Conclusions
This project shows how minor enhancemnts such as BatchNormilization and increased depth can significcantly improve digit classification interpreted by the SVHN Dataset. The optimized CNN outperforms the baseline in accuracy and generalization.

The optimized model achieved:
1. Higher accuracy (89%) that the baseline (86%)
2. better generalization across noisy data
3. Reduced error rates on visually similar digits. 

The workflow was fully executed on a single notebook with organized code, structured markdowns and visual outputs, making the project reproducable and extendable.