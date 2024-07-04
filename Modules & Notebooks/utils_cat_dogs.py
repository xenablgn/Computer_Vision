import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def load_data(train_path, test_path):
    """
    Load image filenames and corresponding labels from directories.
    """
    train_filenames = os.listdir(train_path)
    train_labels = [x.split(".")[0] for x in train_filenames]
    data_train = pd.DataFrame({"filename": train_filenames, "label": train_labels})

    X_train, X_temp = train_test_split(data_train, test_size=0.2, stratify=train_labels, random_state=42)
    label_test_val = X_temp['label']
    X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state=42)

    return X_train, X_val, X_test


def setup_generators(train_df, val_df, test_df, train_dir, image_size, batch_size):
    """
    Set up ImageDataGenerators for training, validation, and testing.
    """
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=15,
                                       horizontal_flip=True,
                                       zoom_range=0.2,
                                       shear_range=0.1,
                                       fill_mode='reflect',
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(train_df,
                                                        directory=train_dir,
                                                        x_col='filename',
                                                        y_col='label',
                                                        batch_size=batch_size,
                                                        target_size=(image_size, image_size))

    val_generator = test_datagen.flow_from_dataframe(val_df,
                                                     directory=train_dir,
                                                     x_col='filename',
                                                     y_col='label',
                                                     batch_size=batch_size,
                                                     target_size=(image_size, image_size),
                                                     shuffle=False)

    test_generator = test_datagen.flow_from_dataframe(test_df,
                                                      directory=train_dir,
                                                      x_col='filename',
                                                      y_col='label',
                                                      batch_size=batch_size,
                                                      target_size=(image_size, image_size),
                                                      shuffle=False)

    return train_generator, val_generator, test_generator


def define_model(image_size, image_channel):
    """
    Define a Convolutional Neural Network (CNN) model for image classification.
    """
    model = Sequential()

    # Input Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, image_channel)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Filter Block 1
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Filter Block 2
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Filter Block 3
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
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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


def evaluate_model(model, generator, batch_size):
    """
    Evaluate the trained Keras model using the provided generator.
    """
    loss, acc = model.evaluate(generator, batch_size=batch_size, verbose=0)
    return loss, acc


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


