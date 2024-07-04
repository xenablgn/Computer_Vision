import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import GridSearchCV, cross_val_predict
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from autils import * 

from sklearn.linear_model import LogisticRegression
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import joblib



def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    training_data_digit = pd.read_csv(file_path)
    y = training_data_digit["label"].values
    X = training_data_digit.drop(columns=["label"]).values
    return X, y


def create_fnn_model(input_shape, learning_rate, layer1_neurons, layer2_neurons, layer3_neurons):
    """
    Create a Feedforward Neural Network (FNN) model.
    """
    model = Sequential([
        Dense(layer1_neurons, activation="relu", input_shape=input_shape, name="layer1"), 
        Dense(layer2_neurons, activation="relu", name="layer2"),
        Dense(layer3_neurons, activation="relu", name="layer3"),
        Dense(10, activation="linear", name="layer4")
    ], name="FNN-model")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def create_cnn_model(input_shape, learning_rate=None, layer1_neurons=None, layer2_neurons=None):
    """
    Create a Convolutional Neural Network (CNN) model.
    """
    model = Sequential([
        Conv2D(layer1_neurons, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(layer2_neurons, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='linear')
    ], name="CNN-model")
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """
    Evaluate the performance of the model on training and validation data.
    """
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    train_preds = np.argmax(model.predict(X_train), axis=1)
    train_f1 = f1_score(y_train, train_preds, average='macro')
    train_recall = recall_score(y_train, train_preds, average='macro')

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    val_preds = np.argmax(model.predict(X_val), axis=1)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    val_recall = recall_score(y_val, val_preds, average='macro')

    return {'accuracy': train_accuracy, 'f1': train_f1, 'recall': train_recall}, {'accuracy': val_accuracy, 'f1': val_f1, 'recall': val_recall}


def cross_validate_fnn(X, y, n_splits, n_epochs, param_grid):
    """
    Perform cross-validation with grid search to find the best FNN model hyperparameters.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_f1_score = 0  # Initialize best F1 score
    best_model = None
    best_hyperparameters = None
    best_train_metrics = {'accuracy': [], 'f1': [], 'recall': []}
    best_val_metrics = {'accuracy': [], 'f1': [], 'recall': []}
    training_epoch_metrics = {'accuracy': [], 'f1': [], 'recall': []}
    validation_epoch_metrics = {'accuracy': [], 'f1': [], 'recall': []}

    learning_rates = param_grid['learning_rate']
    layer1_sizes = param_grid['layer1_neurons']
    layer2_sizes = param_grid['layer2_neurons']
    layer3_sizes = param_grid['layer3_neurons']

    for lr in learning_rates:
        for layer1_neurons in layer1_sizes:
            for layer2_neurons in layer2_sizes:
                for layer3_neurons in layer3_sizes:
                    print(f"Testing FNN model with learning rate {lr}, {layer1_neurons} neurons in the first layer, {layer2_neurons} neurons in the second layer, and {layer3_neurons} neurons in the third layer")
                    fold_train_metrics = {'accuracy': [], 'f1': [], 'recall': []}
                    fold_val_metrics = {'accuracy': [], 'f1': [], 'recall': []}

                    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
                        X_train, X_val = X[train_index], X[val_index]
                        y_train, y_val = y[train_index], y[val_index]

                        model = create_fnn_model(input_shape=(X_train.shape[1],), learning_rate=lr, 
                                                 layer1_neurons=layer1_neurons, layer2_neurons=layer2_neurons, layer3_neurons=layer3_neurons)
                        
                        for epoch in range(1, n_epochs + 1):
                            print(f"Fold {fold}, Epoch {epoch}")
                            model.fit(X_train, y_train, epochs=1, verbose=0)
                            train_metrics, val_metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

                            for metric in ['accuracy', 'f1', 'recall']:
                                fold_train_metrics[metric].append(train_metrics[metric])
                                fold_val_metrics[metric].append(val_metrics[metric])
                                
                                training_epoch_metrics[metric].append(train_metrics[metric])
                                validation_epoch_metrics[metric].append(val_metrics[metric])

                            if np.mean(fold_val_metrics['f1']) > best_f1_score:
                                best_f1_score = np.mean(fold_val_metrics['f1'])
                                best_model = model
                                best_hyperparameters = {'learning_rate': lr, 'layer1_neurons': layer1_neurons, 'layer2_neurons': layer2_neurons, 'layer3_neurons': layer3_neurons}
                                best_train_metrics = fold_train_metrics
                                best_val_metrics = fold_val_metrics

    print(f"Best FNN model found with hyperparameters: {best_hyperparameters}")
    print(f"Best F1 Score: {best_f1_score}")

    return best_model, best_hyperparameters, training_epoch_metrics, validation_epoch_metrics, best_train_metrics, best_val_metrics


def cross_validate_cnn(X, y, n_splits, n_epochs, param_grid, input_shape= None):
    """
    Perform cross-validation with grid search to find the best CNN model hyperparameters.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_f1_score = 0  # Initialize best F1 score
    best_model = None
    best_hyperparameters = None
    best_train_metrics = {'accuracy': [], 'f1': [], 'recall': []}
    best_val_metrics = {'accuracy': [], 'f1': [], 'recall': []}
    training_epoch_metrics = {'accuracy': [], 'f1': [], 'recall': []}
    validation_epoch_metrics = {'accuracy': [], 'f1': [], 'recall': []}

    learning_rates = param_grid['learning_rate']
    layer1_sizes = param_grid['layer1_neurons']
    layer2_sizes = param_grid['layer2_neurons']

    for lr in learning_rates:
        for layer1_neurons in layer1_sizes:
            for layer2_neurons in layer2_sizes:
                print(f"Testing CNN model with learning rate {lr}, {layer1_neurons} neurons in the first layer, and {layer2_neurons} neurons in the second layer")
                fold_train_metrics = {'accuracy': [], 'f1': [], 'recall': []}
                fold_val_metrics = {'accuracy': [], 'f1': [], 'recall': []}

                for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
                    X_cnn = X.reshape(-1, *input_shape) / 255.0  # Use input_shape here
                    X_train, X_val = X_cnn[train_index], X_cnn[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    model = create_cnn_model(input_shape=input_shape, learning_rate=lr, 
                                             layer1_neurons=layer1_neurons, layer2_neurons=layer2_neurons)
                    
                    for epoch in range(1, n_epochs + 1):
                        print(f"Fold {fold}, Epoch {epoch}")
                        model.fit(X_train, y_train, epochs=1, verbose=0)
                        train_metrics, val_metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

                        for metric in ['accuracy', 'f1', 'recall']:
                            fold_train_metrics[metric].append(train_metrics[metric])
                            fold_val_metrics[metric].append(val_metrics[metric])
                            
                            training_epoch_metrics[metric].append(train_metrics[metric])
                            validation_epoch_metrics[metric].append(val_metrics[metric])

                        if np.mean(fold_val_metrics['f1']) > best_f1_score:
                            best_f1_score = np.mean(fold_val_metrics['f1'])
                            best_model = model
                            best_hyperparameters = {'learning_rate': lr, 'layer1_neurons': layer1_neurons, 'layer2_neurons': layer2_neurons}
                            best_train_metrics = fold_train_metrics
                            best_val_metrics = fold_val_metrics

    print(f"Best CNN model found with hyperparameters: {best_hyperparameters}")
    print(f"Best F1 Score: {best_f1_score}")

    return best_model, best_hyperparameters, training_epoch_metrics, validation_epoch_metrics, best_train_metrics, best_val_metrics


def save_model_cnn_fnn(model, filename, folder="../Models/"):
    """
    Save the trained model to a file.
    """
    model.save(folder + filename)
    print(f"Model saved as {folder + filename}")

def load_model_cnn_fnn(filename, folder="../Models/"):
    """
    Load a saved model from a file.
    """
    # Load model
    model_filename = os.path.join(folder, filename)
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file '{model_filename}' not found.")

    loaded_model = load_model(model_filename)
    print(f"Model loaded from {model_filename}")

    return loaded_model

def plot_metrics(training_metrics, validation_metrics, n_folds, n_epochs):
    """
    Plot the training and validation metrics for each fold.
    """
    epochs = range(1, n_epochs + 1)
    plt.figure(figsize=(15, 5))
    for fold in range(n_folds):
        plt.subplot(1, n_folds, fold + 1)
        for metric in ['accuracy', 'f1', 'recall']:
            plt.plot(epochs, training_metrics[metric][fold * n_epochs:(fold + 1) * n_epochs], label=f'Training {metric.capitalize()}')
            plt.plot(epochs, validation_metrics[metric][fold * n_epochs:(fold + 1) * n_epochs], label=f'Validation {metric.capitalize()}')
        plt.title(f"Fold {fold + 1} Metrics")
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
    plt.tight_layout()
    plt.show()


def print_best_model_performance(best_train_metrics, best_val_metrics):
    """
    Print the average best model performance metrics.
    """
    avg_best_train_metrics = {metric: np.mean(best_train_metrics[metric]) for metric in best_train_metrics.keys()}
    avg_best_val_metrics = {metric: np.mean(best_val_metrics[metric]) for metric in best_val_metrics.keys()}
    print("Best Model Training Metrics- Average of k-folds :")
    for metric, value in avg_best_train_metrics.items():
        print(f"{metric.capitalize()}: {value}")
    print("\nBest Model Validation Metrics - Average of k-folds :")
    for metric, value in avg_best_val_metrics.items():
        print(f"{metric.capitalize()}: {value}")



def predict_fnn(model, X_test, y_test, num_samples):
    """
    Make predictions using the trained FNN model on the test data and calculate various metrics.
    """
    logits = model.predict(X_test)
    predictions = tf.nn.softmax(logits)
    
    # Calculate predicted classes
    predicted_classes = np.argmax(predictions, axis=1)
    
   # Calculate accuracy
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Accuracy on Test Data: {accuracy}")
    
    print("Predictions for Test Data:")
    for i in range(num_samples):
        print(f"Sample {i + 1} - True label: {y_test[i]}, Predicted: {predicted_classes[i]}, Probabilities: {predictions[i]}")
    
    return predictions,accuracy



def predict_cnn(model, X_test, y_test, num_samples, input_shape=None):
    """
    Make predictions using the trained CNN model on the test data and calculate accuracy.
    """
    X_test_cnn = X_test.reshape(-1, *input_shape) / 255.0

    logits = model.predict(X_test_cnn)
    predictions = tf.nn.softmax(logits)
    
    predicted_classes = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Accuracy on Test Data: {accuracy}")
    
    print("Predictions for Test Data:")
    for i in range(num_samples):
        print(f"Sample {i + 1} - True label: {y_test[i]}, Predicted: {predicted_classes[i]}, Probabilities: {predictions[i]}")
    
    return predictions, accuracy


def train_best_logistic_regression_model(X, y, param_grid, save_path="../Models/best_logistic_regression_model.pkl"):
    """
    Train the best logistic regression model using grid search with cross-validation and save it to a file.
    Print the best parameters found during the grid search.
    """
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_model = LogisticRegression(solver='lbfgs', max_iter=1000)  # Increase max_iter for better convergence

    grid_search = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=5, scoring='accuracy', return_train_score=True)
    grid_search.fit(X_scaled, y)

    best_model = grid_search.best_estimator_

    # Print the best parameters
    print("Best parameters found:", grid_search.best_params_)

    joblib.dump(best_model, save_path)
    return best_model


def predict_logistic(model, X_test, y_test, num_samples):
    """
    Predict labels using logistic regression model and print sample predictions.
    Also compute and print evaluation metrics: accuracy, precision, recall, and F1 score.
    """
    predictions = model.predict(X_test)

    print("Predictions for Test Data:")
    for i in range(num_samples):
        print(f"Sample {i + 1} - True label: {y_test[i]}, Predicted: {predictions[i]}")

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    # Print evaluation metrics
    print(f"\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")

    return predictions, X_test[:num_samples]


def plot_predictions(predictions, true_labels, num_samples):
    """
    Plot predicted probabilities vs. true labels.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(num_samples), predictions[:num_samples], color='blue', label='Predicted')
    plt.scatter(range(num_samples), true_labels[:num_samples], color='red', label='True')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.title('Predicted Labels vs. True Labels')
    plt.legend()
    plt.grid(True)
    plt.show()


def load_best_logistic_regression_model(model_path="../Models/best_logistic_regression_model.pkl"):
    """
    Load the best logistic regression model from a saved file.
    """
    loaded_model = joblib.load(model_path)
    return loaded_model
