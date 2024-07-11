import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import plotly.express as px
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import confusion_matrix
import itertools
from utils_const import *

np.random.seed(42)
tf.random.set_seed(42)


def create_dataframe(root_path):
    """
    Function to create DataFrame from directory structure.
    """
    images = []
    labels = []

    for class_name in os.listdir(root_path):
        if class_name not in CLASS_NAMES:
            continue

        class_label = CLASS_NAMES_LABEL[class_name]
        class_path = os.path.join(root_path, class_name)

        for filename in os.listdir(class_path):
            images.append(os.path.join(class_name, filename))
            labels.append(str(class_label))  # Convert label to string

    return pd.DataFrame({"filename": images, "label": labels})


def split_train_val(train_df, test_df, validation_split=0.2):
    """
    Split train_df into train and validation DataFrames.
    """
    train_df, val_df = train_test_split(train_df, test_size=validation_split, stratify=train_df['label'])
    return train_df, val_df, test_df


def create_image_generators(train_df, val_df, test_df, image_size, batch_size):
    """
    Create ImageDataGenerators for training, validation, and test sets.
    """
    # Training ImageDataGenerator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Test ImageDataGenerator without augmentation
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Training generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=TRAIN_PATH,
        x_col='filename',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True
    )

    # Validation generator
    val_generator = train_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=TRAIN_PATH,
        x_col='filename',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False  # No need to shuffle validation data
    )

    # Test generator
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=TEST_PATH,
        x_col='filename',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False  # No need to shuffle test data
    )

    return train_generator, val_generator, test_generator


def define_model(image_size, image_channel):
    """
    Define a Convolutional Neural Network (CNN) model for image classification.
    """
    model = Sequential()

    # Input Layer
    model.add(Input(shape=(image_size[0], image_size[1], image_channel)))  # Input layer to specify input shape

    # Convolutional Layers
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Fully Connected Dense Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(NB_CLASSES, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, train_generator, val_generator, epochs, callbacks):
    """
    Train the defined Keras model using the provided generators.
    """
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        callbacks=callbacks,
                        epochs=epochs)
    return history


def plot_training_history(history):
    """
    Plot training history (loss and accuracy) over epochs using Matplotlib and Seaborn.
    """
    error = pd.DataFrame(history.history)

    plt.figure(figsize=(18, 5), dpi=200)
    sns.set_style('darkgrid')

    plt.subplot(121)
    plt.title('Cross Entropy Loss', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.plot(error['loss'], label='Training Loss')
    plt.plot(error['val_loss'], label='Validation Loss')
    plt.legend()

    plt.subplot(122)
    plt.title('Classification Accuracy', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.plot(error['accuracy'], label='Training Accuracy')
    plt.plot(error['val_accuracy'], label='Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_model(model, model_save_path):
    """
    Save the trained Keras model to the specified path.
    """
    model.save(model_save_path)


def evaluate_few_batches(model, test_generator, batch_size, num_batches=3):
    """
    Evaluate the model on a few batches of data from the test generator.
    """
    total_loss = 0.0
    total_accuracy = 0.0
    true_labels = []
    predicted_labels = []
    images = []
    
    test_generator.on_epoch_end()  # This will shuffle the indices of the generator
    
    shuffled_indices = list(range(len(test_generator)))
    random.shuffle(shuffled_indices)
    random_indices = shuffled_indices[:num_batches]
    
    for idx in random_indices:
        x_batch, y_batch = test_generator[idx]
        
        batch_loss, batch_accuracy = model.evaluate(x_batch, y_batch, batch_size=batch_size, verbose=0)
        
        total_loss += batch_loss
        total_accuracy += batch_accuracy
        
        predictions = model.predict(x_batch, batch_size=batch_size)
        predicted_classes = np.argmax(predictions, axis=1)
        
        if y_batch.ndim > 1:  # Check if y_batch is one-hot encoded
            true_labels.extend(np.argmax(y_batch, axis=1).tolist())
        else:
            true_labels.extend(y_batch.tolist())
        
        predicted_labels.extend(predicted_classes)
        
        images.extend(x_batch)
    
    average_loss = total_loss / num_batches
    average_accuracy = total_accuracy / num_batches
    
    return average_loss, average_accuracy, true_labels, predicted_labels, images


def plot_evaluation_images(images, true_labels, predicted_labels, class_names):
    """
    Plot images with their true and predicted labels.
    """
    num_samples = len(images)
    
    plt.figure(figsize=(15, 3 * (num_samples // 5 + 1)))  # Adjust height based on number of rows
    
    for i in range(num_samples):
        plt.subplot((num_samples // 5) + 1, min(num_samples, 5), i + 1)
        plt.imshow(images[i])
        
        true_label_idx = int(true_labels[i])  # Convert to int for indexing
        pred_label_idx = int(predicted_labels[i])  # Convert to int for indexing
        
        plt.title(f'True: {class_names[true_label_idx]}\nPredicted: {class_names[pred_label_idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_exp(expression):
    """
    Plot images from a specified expression category.
    """
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 12))
    for i in range(1, 10):  # Loop through 1 to 9
        plt.subplot(3, 3, i)
        img_path = os.path.join(FOLDER_PATH, "train", expression, os.listdir(os.path.join(FOLDER_PATH, "train", expression))[i])
        img = load_img(img_path, target_size=(PIC_SIZE, PIC_SIZE))
        plt.imshow(img)
    plt.show()


def walk_data(mydict, folder_path):
    """
    Walk through a directory and count files in each subdirectory.
    """
    for (root, dirs, files) in os.walk(folder_path, topdown=False):
        if len(files) > 0:
            mydict[root] = len(files)


def fix_keys(mydict):
    """
    Fix dictionary keys by extracting the last part of the path.
    """
    new_dict = {k.split('/')[-1]: v for k, v in mydict.items()}
    return new_dict


def plot_dist(my_dict, data):
    """
    Plot a horizontal bar chart to visualize data distribution.
    """
    fig = px.bar(x=list(my_dict.values()),
                 y=list(my_dict.keys()),
                 orientation='h',
                 color=list(my_dict.values()),
                 title=data + ' Distribution',
                 opacity=0.8,
                 color_discrete_sequence=px.colors.diverging.curl,
                 template='plotly_dark'
                )
    fig.update_xaxes()
    fig.show()


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()