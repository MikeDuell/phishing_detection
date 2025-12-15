import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def evaluate_model(y_true, y_pred):
    y_true_oh = pd.get_dummies(y_true)  # One-hot encode true labels
    y_pred_oh = pd.get_dummies(y_pred)  # One-hot encode predicted labels
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovr')
    }

# Step 10: Define function to plot evaluation metrics
def plot_metrics(name, metrics):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(f"{name} Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    for p in ax.patches:
        value = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.,  # x position (center of bar)
            value + 0.01,  # y position (slightly above bar)
            f'{value:.2f}',  # formatted text
            ha='center', va='bottom'  # alignment
        )

    fig.tight_layout()
    plt.show()
    return fig

def display_supervised_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=['Benign', 'Phishing'], yticklabels=['Benign', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.show()

def run_supervised_models(X_train_scaled, X_test_scaled,  y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    results = {}  # Store evaluation results
    predictions = {}  # Store predictions
    trained_models = {}
    # Loop through each classical model
    for name, model in models.items():
        m = model.fit(X_train_scaled, y_train)  # Train model
        y_pred = model.predict(X_test_scaled)  # Predict on test set
        predictions[name] = y_pred
        print(f"\n{name} Classification Report:\n")
        print(classification_report(y_test, y_pred, zero_division=0))  # Print detailed report
        metrics = evaluate_model(y_test, y_pred)  # Compute metrics
        plot_metrics(name, metrics)  # Plot metrics
        display_supervised_confusion(y_test, y_pred, name)
        trained_models[name] = m

