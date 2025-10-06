# Ahmed Negm | 501101640
# AER 850 Assignment 1
# To view the Prediction of the model (step 7) please run the file: predict_step.py

#Importing required libraries and modules to do the assignment
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from pathlib import Path



# Question 1 - Load and process data
file_path = r'data/Project_Data.csv'

def load_and_process_data(file_path):
    """
    This function loads the CSV data, converts it into a Pandas DataFrame, and 
    ensures the data is ready for analysis by returning a cleaned DataFrame.
    """
    data = pd.read_csv(file_path)
    #file_path = Path("data") / "Project_Data.csv"
    return data

data = load_and_process_data(file_path)
print("Data loaded and processed:")
#print the first few rows to confirm
print(data.head()) 

# Question 2 - Visualizing data
def visualize_data(data):
    print("\nStatistical Summary of the Data:")
    # printing the statistical summary of the data
    print(data.describe())  

    # plotting histogram for X coordinates
    plt.figure(figsize=(6, 4))
    plt.hist(data['X'], bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of X Coordinates')
    plt.xlabel('X')
    plt.ylabel('Frequency')
    plt.show()

    # plotting histogram for Y coordinates
    plt.figure(figsize=(6, 4))
    plt.hist(data['Y'], bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Y Coordinates')
    plt.xlabel('Y')
    plt.ylabel('Frequency')
    plt.show()

    #plotting histogram for Z coordinates
    plt.figure(figsize=(6, 4))
    plt.hist(data['Z'], bins=20, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Z Coordinates')
    plt.xlabel('Z')
    plt.ylabel('Frequency')
    plt.show()

visualize_data(data)


# Question 3 - Correlation Analysis
def correlation_analysis(data):
    correlation_matrix = data.corr()
    print("\nCorrelation Matrix:")
    # print the correlation matrix
    print(correlation_matrix)  

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix (Features vs Target)")
    plt.show()

correlation_analysis(data)


# Question 4 - Train classification models
def train_classification_models(data):
    # these are the features we are using
    X = data[['X', 'Y', 'Z']]  
    # 'Step' is the  target column name
    y = data['Step']  
    
    #random state 42 is to make sure the code produce the same result for the random operations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10]}
    
    svc_model = SVC(random_state=42)
    svc_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    
    knn_model = KNeighborsClassifier()
    knn_param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
    
    # This is GridSearchCV for Random Forest
    grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=5, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_rf_model = grid_search_rf.best_estimator_
    print(f"Best Random Forest Params: {grid_search_rf.best_params_}")

    # This is GridSearchCV for SVC (Support Vector Machine)
    grid_search_svc = GridSearchCV(estimator=svc_model, param_grid=svc_param_grid, cv=5, n_jobs=-1)
    grid_search_svc.fit(X_train, y_train)
    best_svc_model = grid_search_svc.best_estimator_
    print(f"Best SVC Params: {grid_search_svc.best_params_}")

    #this is GridSearchCV for KNN (K-Nearest Neighbors)
    grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=knn_param_grid, cv=5, n_jobs=-1)
    grid_search_knn.fit(X_train, y_train)
    best_knn_model = grid_search_knn.best_estimator_
    print(f"Best KNN Params: {grid_search_knn.best_params_}")

    #this is RandomizedSearchCV for Random Forest
    random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42)
    random_search_rf.fit(X_train, y_train)
    best_rf_random_model = random_search_rf.best_estimator_
    print(f"Best Randomized RF Params: {random_search_rf.best_params_}")

    # this is the Model Evaluation (Using Test Set)
    models = [best_rf_model, best_svc_model, best_knn_model, best_rf_random_model]
    model_names = ['Random Forest', 'SVC', 'KNN', 'Random Forest (RandomizedSearch)']

    for name, model in zip(model_names, models):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Test Accuracy: {accuracy:.4f}")
    
    return best_rf_model, best_svc_model, best_knn_model, best_rf_random_model, X_train, X_test, y_train, y_test


# Question 5 - Model Performance Analysis
def evaluate_model_performance(models, model_names, X_test, y_test):
    performance_metrics = {}

    for name, model in zip(model_names, models):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted') 
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        performance_metrics[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        #printing the metrics for each model
        print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        #plotting the confusion matrix for each model
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    return performance_metrics

#calling the function to train the models and get the best ones
best_rf_model, best_svc_model, best_knn_model, best_rf_random_model, X_train, X_test, y_train, y_test = train_classification_models(data)


#calling the evaluation function to analyze model performance
models = [best_rf_model, best_svc_model, best_knn_model, best_rf_random_model]
model_names = ['Random Forest', 'SVC', 'KNN', 'Random Forest (RandomizedSearch)']

# evaluating the models using the function defined earlier
evaluate_model_performance(models, model_names, X_test, y_test)




def stacked_model_performance_analysis(X_train, X_test, y_train, y_test, best_rf_model, best_svc_model):
    """
    This function uses scikit-learn's StackingClassifier to combine two base models (Random Forest, SVC),
    and evaluates the stacked model based on accuracy, precision, recall, F1 score, and confusion matrix.
    
    Parameters:
    - X_train, X_test: Training and testing features
    - y_train, y_test: Training and testing target labels
    - best_rf_model, best_svc_model: Pre-trained models to be used as base models for stacking
    
    Returns:
    - performance_metrics: A dictionary containing performance metrics (accuracy, precision, recall, f1)
    """
    # Define the base models
    base_models = [
        ('rf', best_rf_model),  # Random Forest
        ('svc', best_svc_model)  # Support Vector Classifier
    ]
    
    # estimator
    final_estimator = LogisticRegression(max_iter=200)

    # Create the Stacking Classifier
    stacked_model = StackingClassifier(estimators=base_models, final_estimator=final_estimator)

    # training the stacked model
    stacked_model.fit(X_train, y_train)

    # predicting with the stacked model
    y_pred_stacked = stacked_model.predict(X_test)

    # calculating performance metrics
    accuracy_stacked = accuracy_score(y_test, y_pred_stacked)
    precision_stacked = precision_score(y_test, y_pred_stacked, average='weighted')
    recall_stacked = recall_score(y_test, y_pred_stacked, average='weighted')
    f1_stacked = f1_score(y_test, y_pred_stacked, average='weighted')

    # printing performance metrics for the stacked model
    print(f"Stacked Model Accuracy: {accuracy_stacked:.4f}")
    print(f"Stacked Model Precision: {precision_stacked:.4f}")
    print(f"Stacked Model Recall: {recall_stacked:.4f}")
    print(f"Stacked Model F1 Score: {f1_stacked:.4f}")

    # this is the confusion Matrix for Stacked Model
    cm_stacked = confusion_matrix(y_test, y_pred_stacked)

    # plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_stacked, annot=True, fmt='d', cmap='Blues', xticklabels=stacked_model.classes_, yticklabels=stacked_model.classes_)
    plt.title('Stacked Model Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Store the performance metrics in a dictionary for good presentation
    performance_metrics = {
        'Accuracy': accuracy_stacked,
        'Precision': precision_stacked,
        'Recall': recall_stacked,
        'F1 Score': f1_stacked
    }

    return performance_metrics


performance_metrics = stacked_model_performance_analysis(X_train, X_test, y_train, y_test, best_rf_model, best_svc_model)

print("Stacked Model Performance Metrics:")
for metric, value in performance_metrics.items():
    print(f"{metric}: {value:.4f}")
    


# this function will save the model.
def save_model(model, model_name='model.joblib'):
    """
    This function saves the trained model to a joblib file.
    
    Parameters:
    - model: The trained model to be saved
    - model_name: The name of the file where the model will be saved (default is 'model.joblib')
    """
    joblib.dump(model, model_name)
    print(f"Model saved as {model_name}")

# Save the SVC model after training
save_model(best_svc_model, 'best_svc_model.joblib')



