# ============================================================
# modelling.py — untuk MLflow Project CI
# Versi ini dirancang untuk berjalan di GitHub Actions
# ============================================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Konfigurasi ──────────────────────────────────────────────
DATA_PATH    = 'student_performance_preprocessing.csv'
TARGET_COL   = 'GradeClass'
RANDOM_STATE = 42

def load_data(path, target_col):
    df       = pd.read_csv(path)
    df_train = df[df['split'] == 'train'].drop(columns=['split'])
    df_test  = df[df['split'] == 'test'].drop(columns=['split'])
    X_train  = df_train.drop(columns=[target_col])
    y_train  = df_train[target_col]
    X_test   = df_test.drop(columns=[target_col])
    y_test   = df_test[target_col]
    print(f"✅ Data: Train={X_train.shape} Test={X_test.shape}")
    return X_train, X_test, y_train, y_test

def plot_confusion_matrix(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_ylabel('Aktual')
    ax.set_xlabel('Prediksi')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    return 'confusion_matrix.png'

def main():
    # Tracking URI dari environment variable
    # Di GitHub Actions akan di-set otomatis
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', './mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('student-performance-ci')

    X_train, X_test, y_train, y_test = load_data(
        DATA_PATH, TARGET_COL
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name='CI_RandomForest'):
        model = RandomForestClassifier(
            n_estimators = 100,
            max_depth    = 20,
            random_state = RANDOM_STATE
        )
        model.fit(X_train, y_train)

        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,
                                    average='weighted',
                                    zero_division=0)
        recall    = recall_score(y_test, y_pred,
                                 average='weighted',
                                 zero_division=0)
        f1        = f1_score(y_test, y_pred,
                             average='weighted',
                             zero_division=0)

        # Manual log tambahan
        mlflow.log_metric('test_accuracy',  acc)
        mlflow.log_metric('test_precision', precision)
        mlflow.log_metric('test_recall',    recall)
        mlflow.log_metric('test_f1',        f1)

        cm_path = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_artifact(cm_path)

        print(f"\n📊 HASIL CI RUN:")
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {precision:.4f}")
        print(f"   Recall    : {recall:.4f}")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"\n✅ Training selesai")

if __name__ == '__main__':
    main()