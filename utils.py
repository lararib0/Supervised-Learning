from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Train test split, SMOTE no treino
def train_split(X, y, random_state=None):
    if random_state is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test


# Train, predict and evaluate
def test_evaluate(model, X, y, random_state=None):
    X_train, X_test, y_train, y_test = train_split(X, y, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test


# Plot dos histogramas da função 'evaluate_function' para cada modelo
def plot_histograms(models_dict, evaluate_function, X, y):
    fig, axs = plt.subplots(1, len(models_dict), figsize=(18, 5))

    for model_name, model_dict in models_dict.items():
        accuracy = evaluate_function(model_dict['model'], X,
                                     y)  # Evaluating the model using the provided evaluation function

        # Plotting the histogram
        ax = axs[list(models_dict.keys()).index(model_name)]
        ax.hist(accuracy, bins=10, edgecolor='black')
        ax.set_title(f"{model_name} Accuracy score: {np.mean(accuracy):.2f}")
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# Avaliação do modelo com o método hold-out
def hold_out_evaluate(model, X, y, num_it=50):
    accuracies = []
    for i in range(num_it):
        y_pred, y_test = test_evaluate(model, X, y)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    return accuracies


# Avaliação do modelo com o método cross-validation
def cross_val_evaluate(model, X, y, cv=10):
    return cross_val_score(model, X, y, cv=cv, scoring='accuracy')


# Plot da matriz de confusão
def confusion_matrix_plot(models_dict, X, y):
    fig, axes = plt.subplots(1, len(models_dict), figsize=(5 * len(models_dict), 5))  # Create subplots
    for idx, (model_name, model_dict) in enumerate(models_dict.items()):
        y_pred, y_test = test_evaluate(model_dict['model'], X, y, random_state=42)
        cm = confusion_matrix(y_test, y_pred)

        # Plotting the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[idx], cmap='Blues')
        axes[idx].title.set_text(model_name)
        # Turn off grid lines
        axes[idx].grid(False)

        # Remove the lines in the center
        for spine in axes[idx].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.show()


# print do relatório da classificação
def report(models, features, label):
    reports = []
    for model_name, model_dict in models.items():
        print(f"Model: {model_name}")
        model = model_dict['model']
        y_pred, y_test = test_evaluate(model, features, label, random_state=42)
        rep = classification_report(y_pred, y_test)
        print(rep)
        reports.append(rep)

# Split do dataset em features e label e encoding das features categóricas
def split(df, columns_to_drop=None):
    if columns_to_drop is None:
        columns_to_drop = ['Class']

    features = df.drop(columns=columns_to_drop)
    label = df[columns_to_drop[0]]
    categorical_features = features.select_dtypes(include=[object, 'category']).columns.tolist()
    # Impute categorical features using the most frequent value
    most_frequent_imputer = SimpleImputer(strategy='most_frequent')
    features[categorical_features] = most_frequent_imputer.fit_transform(features[categorical_features])

    features = pd.get_dummies(features, columns=categorical_features)
    return features, label
