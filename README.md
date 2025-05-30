# BIG-DATA-CW2
## Classifying digits from (0-9) from a 32x32RGB image using a loaded CNN model.

**Student ID'S**: Mike Shier **(UP2127137)**, Jack Elliott **(UP2119254)** and 
Abdelrahaman Elshafei
**(UP2088294)**

## Introduction
This project focuses on building a complete machine learning (ML) pipeline for classifying digits (0–9) using a Convolutional Neural Network (CNN). The dataset used is the Street View House Numbers (SVHN), which consists of 32x32 RGB images of digits cropped from natural scene images. The objective is to deploy a reliable digit classification model with a strong emphasis on pipeline development, evaluation, and deployment of CNNs.

## Business Objectives
The goal is to accurately classify individual digits from real-world image data. High classification accuracy is essential, as such a system could be applied in real-time digit recognition scenarios like address or license plate recognition. The target is to achieve over 90% accuracy on unseen test data.

## ML Pipeline

### 1. Data Collection
The project begins by importing two foundational Python modules: os and urllib.request. The os module provides functionality to create and manage directories in a way that works across different operating systems, which is important for organizing and accessing the dataset. The urllib.request module enables the script to make HTTP requests, which is essential for downloading the SVHN dataset files directly from Stanford's server. This approach ensures the entire pipeline is reproducible and does not require manual data setup.

            import os
            import urllib.request

To organize the downloaded data, the script creates a new directory called 'data' using the os.makedirs() function. By passing the parameter exist_ok=True, the code avoids throwing an error if the folder already exists. This design makes the script idempotent, allowing it to be safely re-run without duplicating effort or interrupting the workflow due to existing directory conflicts. 

            # Creating a folder for the data to be handled
            os.makedirs("data", exist_ok = True)

The script defines a dictionary of named URLs that contain the download links for three subsets of the SVHN dataset: training, testing, and extra data. Each key in the dictionary corresponds to a specific subset and maps to the appropriate URL. This method allows for easy iteration during the download process and keeps the dataset sources well-organized and centralized within the code, making updates or changes simple and scalable.

            # Defining the URLs for training and testing the data
            urls = {
                "train":"http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                "test":"http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                "extra":"http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
            }

To automate the retrieval of the SVHN data, a function called download_svhn() is defined. This function loops through each item in the URLs dictionary and checks whether the corresponding dataset file already exists in the data directory. If a file is missing, it uses urllib.request.urlretrieve() to download it and saves it with a name like train_32x32.mat. Progress messages are printed during each download to provide user feedback. This function streamlines the initial setup and ensures that all required datasets are locally available before further processing.


            # Downloading functions
            def download_svhn():
                for name, url in urls.items():
                    filepath = f"data/{name}_32x32.mat"
                    if not os.path.exists(filepath):
                        print(f"Downloading {name}data...")
                        urllib.request.urlretrieve(url,filepath)
                        print(f"{name.capitalize()}Data has been download sucessfully.")
                    else:
                        print(f"{name.capitalize()} Data already exists.")

            download_svhn()

            print("All SVHN files are curently available and have been downloaded.")

Once these files are downloaded, a pathway is created for each of the .mat files using the f string: f"data/{name}_32x32.mat". The file sizes are calculated for user visualisation through os.path.getsize(file_path), which would originally return the size in Bytes. But, it is converted for the user in MB by 1024**2 which is 

            for name in urls:
                file_path = f"data/{name}_32x32.mat"
                print(f"{name} File size:{os.path.getsize(file_path) / (1024**2):.2f} MB")

Several critical libraries are imported next to support data handling, visualization, and machine learning. The scipy.io module is used to read .mat files, which is the format in which the SVHN dataset is provided. numpy is employed for numerical operations and array manipulations. The train_test_split function from sklearn is utilized to divide the dataset into training and validation subsets. For visualizations, matplotlib.pyplot allows the display of images and plots. Lastly, to_categorical from tensorflow.keras.utils is used to convert numeric digit labels into one-hot encoded vectors, a format required by many neural network architectures for classification tasks.

            import scipy.io
            import numpy as np
            from sklearn.model_selection import train_test_split
            import matplotlib.pyplot as plt
            from tensorflow.keras.utils import to_categorical
            import os
            import gc
            import tensorflow as tf

Below can be seen as a debug feature that was implemented to show how big the downloads were for each data set that was installed.

            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

The script then loads the training and testing datasets using scipy.io.loadmat(), which reads MATLAB-formatted .mat files into Python dictionaries. Each dictionary contains an image array and a corresponding label array. For example, train_data['X'] contains the training images, while train_data['y'] holds the labels. These datasets are now ready to be preprocessed and reshaped for use in a convolutional neural network.

            # Loading the data 
            train_data = scipy.io.loadmat("data/train_32x32.mat")
            test_data = scipy.io.loadmat("data/test_32x32.mat")
---
### 2. EDA
Before model training begins, the script prints out the shapes of the training, validation, and test datasets. This serves as a sanity check to ensure that the preprocessing pipeline has correctly formatted the data. It verifies that the inputs have four dimensions (samples, height, width, channels) and that the labels align with the number of samples.

            import gc

            # Transposing image arrays to the correct shape: e.g.(32,32,3)
            X_train = np.transpose(train_data['X'], (3, 0, 1, 2))
            y_train = train_data['y'].flatten()
            y_train[y_train == 10] = 0

            X_test = np.transpose(test_data['X'], (3, 0, 1, 2))
            y_test = train_data['y'].flatten()
            y_test[y_test == 10] = 0


            del train_data, test_data
            gc.collect()

To enhance model performance and ensure consistent input scaling, the pixel values of the image data are normalised. Since the original RGB values range from 0 to 255, they are converted to a floating-point format (float32) and divided by 255.0. This transformation scales the pixel intensities to a range of 0 to 1, which is known to improve convergence speed and training stability in neural networks. 

            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0

Now, the extra data needs to be processed and saved. Similarly to the first step of the EDA, the labels are converted to 0. This is because SVHN uses 10 to represent the digit 0. The batch size is reduced to 1000 to stop the kernel from dying. The extra data is of course converted to float32 aswell. For efficiency in the future, each chunk is saved to the disk in a compressed .npz format. Towards the bottom, the code includes gc.collect() which is a garbage collection, avoiding memory leaks. 

            import gc

            extra_data = scipy.io.loadmat("data/extra_32x32.mat")
            y_extra = extra_data['y'].flatten()
            y_extra[y_extra == 10] = 0

            batch_size = 1000 #reduced from 10,000 to help preevent kernal dying.

            for i in range(0, 50000, batch_size):
                X_batch_raw = extra_data['X'][:, :, :, i:i+batch_size]
                X_batch = np.transpose(X_batch_raw, (3, 0, 1, 2)).astype('float32') / 255.0
                y_batch = y_extra[i:i+batch_size]

                np.savez_compressed(f"data/X_extra_{i}_{i+batch_size}.npz", X_batch)
                np.savez_compressed(f"data/y_extra_{i}_{i+batch_size}.npz", y_batch)

                del X_batch_raw, X_batch, y_batch
                gc.collect()

            del extra_data, y_extra
            gc.collect()

            print(" Extra data saved in compressed chunks.")

The code then needs to be divided into training and validation data. Using train_test_splot from the sklearn.model_selection module, the x and y train and val are split into: 20% validation data, and 80% training data. It helps monitor model performance and ensures that class imbalances do not skew evaluation metrics. random_state=42 makes the split reproducible. 

            # Spitting training sets: 80% training, 20% validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size = 0.2, stratify = y_train, random_state = 42)

The SVHN dataset represents the digit '0' using the label '10', which can be confusing during model training. To correct this, the script replaces every instance of label 10 with 0 in both the training and test label arrays. This step ensures consistency with standard digit labels and avoids logical mismatches during training and evaluation. Afterwards, the code splits up the batch sizes into smaller chunks. Making batch sizes [10,000] for the training, and [2000] for the validation; helps with computer efficiency and reduce potential kernel crashes. 

            y_train_cat = to_categorical(y_train_split,num_classes = 10)
            y_val_cat = to_categorical(y_val_split, num_classes = 10)

            X_train_split = X_train_split[:10000]
            y_train_cat = y_train_cat[:10000]
            X_val_split = X_val_split[:2000]
            y_val_cat = y_val_cat[:2000]

Before model training begins, the script prints out the shapes of the training, validation, and test datasets. This serves as a sanity check to ensure that the preprocessing pipeline has correctly formatted the data. It verifies that the inputs have four dimensions (samples, height, width, channels) and that the labels align with the number of samples.

            # Printing Shapes
            print("Train set shape:", X_train_split.shape, y_train_split.shape)
            print("Validation set shape:", X_val_split.shape, y_val_split.shape)
            print("Testing set shape:", X_test.shape, y_test.shape)

The code visualises five random images from the training dataset using matplotlib. Each image is displayed with its corresponding class label as the title. This step is part of exploratory data analysis (EDA), providing a visual confirmation that the data looks as expected and has been loaded and processed correctly. It also helps identify any anomalies or incorrect labels early in the workflow.

            # Visulising some samples 
            plt.figure(figsize=(12,3), dpi= 120)

            for i in range(5):
                plt.subplot(1, 5, i + 1)
                plt.imshow(X_train_split[i], interpolation='nearest')
                plt.title(f"Label:{y_train_split[i]}", fontsize = 10)
                plt.axis('on')

            plt.suptitle("Sample SVHN Training Images", fontsize = 14)
            plt.tight_layout()
            plt.show()
---
### 3. Model Building
The image data loaded from the .mat files initially has a shape of (32, 32, 3, N), where N is the number of images. However, most deep learning frameworks, including Keras, expect image data to follow the format (N, 32, 32, 3). To accommodate this, the script transposes the image arrays using np.transpose() and adjusts the axis ordering. Additionally, label arrays are flattened from shape (N, 1) to (N,) using flatten(), which simplifies further processing. Notably, there is a small error in this section where y_test is mistakenly assigned from train_data['y'] instead of test_data['y'].

            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

            model = Sequential([
                Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation = 'relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(128, activation = 'relu'),
                Dropout(0.5),
                Dense(10, activation = 'softmax')
            ])

Similar to steps in the EDA, the labels are replaced from 10 to 0 using one-hot encoding. 

            from tensorflow.keras.utils import to_categorical


            y_train_cat = to_categorical(y_train_split, num_classes = 10)
            y_val_cat = to_categorical(y_val_split, num_classes = 10) 

If the validation loss does not improve after a certain amount of consecutive epochs, training will be stopped. This saves computational power and time. In the below code, this consecutive number is set to 5. If the epochs have fluctuating results, and the final outcomes perform worse; the code takes the best performing one. This model is saved as model_opt.keras. 

            from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

            early_stop = EarlyStopping(monitor = 'val_loss', patience = 5)
            checkpoint = ModelCheckpoint("model_opt.keras", save_best_only = True)

Due to its large size, the extra data is processed in batches of 10,000 samples to prevent memory overflow. Each batch is reshaped and normalized in the same way as the primary datasets. This approach allows the extra data to be used effectively without compromising system performance.

            X_train_split = X_train_split[:10000]
            y_train_split = y_train_split[:10000]

The script uses train_test_split() from sklearn.model_selection to divide the original training dataset into two subsets: training and validation. Again, this is a repeated step from the EDA. 

            from tensorflow.keras.utils import to_categorical
            y_train_cat = to_categorical(y_train_split)

Like in EDA, the validation is split into batches of 2000.

            X_val_split = X_val_split[:2000]
            y_val_split = y_val_split[:2000]
            y_val_cat = to_categorical(y_val_split)

#### CNN 1:
An initial Convolutional Neural Network (CNN) is built to obtain baseline results. The code uses the keras sequential model, with the architecture consisting of two convulational layers. Including maxpooling, flatten, dense, and dropout layers. The output is flattened and passed through a dense hidden lyaer with 128 units and two dropout layers, in the hopes of reducing overfitting. The final layer uses 'softmax' to classify the images into a one of ten digit classes (0-9). The model uses 'adam' optimiser, categorical cross entropy as its loss function, and accuracy for the evaluation. The CNN is trained for 10 epochs. 

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

#### CNN 2:
This second CNN is an improved version of the first. It uses similar aspects as the first. However, it utilises a third convulational layer which gives it the capibility to learn more complex features. The dropout rates are slightly adjusted to reduce overfitting. This CNN is trained for 15 epochs using the early stop and checkpoint features to obtain the best results. 

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

This cell repeats the dataset downsampling from cell 16, again slicing the first 10,000 and 2,000 samples for training and validation respectively. This repetition could be a remnant of iterative testing and should ideally be removed or consolidated.

            X_train_split = X_train_split[:10000]
            y_train_cat = y_train_cat[:10000]

            X_val_split = X_val_split[:2000]
            y_val_cat = y_val_cat[:2000]

#### CNN Comparison:
Using matplotlib, we can create a side by side chart comparison of CNN 1 and 2. A figure with two subplots is set up: the first being the validation accuracy of epochs for both models, and the second being the same for validation loss. history_base and history_opt store training logs from each model's .fit() call, and are used to extract the performance metrics. Legends, axis labels, titles, and grids are all set for clarity. tight_layout() makes it so the plots do not overlap, and plt.show() displays the chart.

            import matplotlib.pyplot as plt

            plt.figure(figsize = (12, 5))

            # Accuracy 
            plt.subplot(1, 2, 1)
            plt.plot(history_base.history['val_accuracy'], label = 'Baseline')
            plt.plot(history_opt.history['val_accuracy'], label = 'Optimized')
            plt.title('Validation Accuracy Comparison')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid('on')

            # Loss
            plt.subplot(1, 2, 2)
            plt.plot(history_base.history['val_loss'], label = 'Baseline')
            plt.plot(history_opt.history['val_loss'], label = 'Optimized')
            plt.title('Validation Loss Comparison')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            plt.tight_layout()
            plt.grid('on')
            plt.show()

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