                                                            config.py

import os

class Config:
    DATA_DIR = os.path.join(os.getcwd(), 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(os.getcwd(), 'models')
    NOTEBOOKS_DIR = os.path.join(os.getcwd(), 'notebooks')
    RANDOM_SEED = 42
                                                            main.py

from src.data_processing import load_and_process_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.hyperparameter_tuning import tune_hyperparameters
from src.visualization import visualize_results
from config import Config

def main():
    # Load and process data
    X_train, X_test, y_train, y_test = load_and_process_data(Config.RAW_DATA_DIR)

    # Tune hyperparameters
    best_params = tune_hyperparameters(X_train, y_train)

    # Train model with best hyperparameters
    model = train_model(X_train, y_train, best_params)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Model evaluation metrics: {metrics}")

    # Visualize results
    visualize_results(model, X_test, y_test)

if __name__ == "__main__":
    main()

                                                        src/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import Config

def load_and_process_data(data_dir):
    # Load data
    data = pd.read_csv(data_dir + '/data.csv')

    # Split data into features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=Config.RANDOM_SEED)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

                                                        src/model_training.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from config import Config

def train_model(X_train, y_train, params):
    model_type = params['model_type']

    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=params['n_estimators'], random_state=Config.RANDOM_SEED)
    elif model_type == 'svm':
        model = SVC(C=params['C'], kernel=params['kernel'], random_state=Config.RANDOM_SEED)
    elif model_type == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=params['hidden_layer_sizes'], random_state=Config.RANDOM_SEED)
    elif model_type == 'keras_nn':
        model = Sequential([
            Dense(params['first_layer_size'], activation='relu', input_shape=(X_train.shape[1],)),
            Dense(params['second_layer_size'], activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])
    else:
        raise ValueError("Unsupported model type")

    # Train model
    if model_type == 'keras_nn':
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'])
    else:
        model.fit(X_train, y_train)

    return model

                                                    src/model_evaluation.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    return metrics
                                                src/hyperparameter_tuning.py

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from config import Config

def tune_hyperparameters(X_train, y_train):
    model_type = 'random_forest'  # or 'svm', 'neural_network', 'keras_nn'

    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=Config.RANDOM_SEED)
    elif model_type == 'svm':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        model = SVC(random_state=Config.RANDOM_SEED)
    elif model_type == 'neural_network':
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'alpha': [0.0001, 0.001, 0.01]
        }
        model = MLPClassifier(random_state=Config.RANDOM_SEED)
    else:
        raise ValueError("Unsupported model type for hyperparameter tuning")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_
                                                    src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_results(model, X_test, y_test):
    # Predict on test data
    y_pred = model.predict(X_test)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
                                                    src/utils.py

import os
import joblib
from config import Config

def save_model(model, model_name):
    model_path = os.path.join(Config.MODELS_DIR, model_name)
    joblib.dump(model, model_path)

def load_model(model_name):
    model_path = os.path.join(Config.MODELS_DIR, model_name)
    return joblib.load(model_path)

                                                    requirements.txt

pandas
scikit-learn
tensorflow
matplotlib
seaborn
joblib