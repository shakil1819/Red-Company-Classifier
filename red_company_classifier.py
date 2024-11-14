import fire
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score, 
    balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


def save_model(model, out_folder: str):
    """
    Serialise the model to an output folder 
    """
    os.makedirs(out_folder, exist_ok=True)
    model_path = os.path.join(out_folder, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model with adjusted threshold
    """
    y_pred = model.predict_proba(X_test)[:, 1]
    
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_threshold = 0.5
    best_balanced_accuracy = 0
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        balanced_acc = balanced_accuracy_score(y_test, y_pred_binary)
        if balanced_acc > best_balanced_accuracy:
            best_balanced_accuracy = balanced_acc
            best_threshold = threshold
    
    y_pred_binary = (y_pred >= best_threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'average_precision': average_precision_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_binary)
    }
    
    confusion = confusion_matrix(y_test, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
    disp.plot()
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    disp.ax_.set_xticklabels(['Not red', 'Red'])
    disp.ax_.set_yticklabels(['Not red', 'Red'])
    plt.title('Confusion Matrix')
    plt.show()
    
    print('-'*50)
    for metric_name, value in metrics.items():
        print(f'{metric_name}: {value:0.4f}')
    
    return metrics

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Generate data splits. You may create train, validation(optional), test data.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(df1, df2, df3):
    """
    Preprocess the dataframes before training
    """
    categorical_cols = ['ds1_f_10', 'ds1_f_38', 'ds1_f_41']
    for col in categorical_cols:
        df1[col] = LabelEncoder().fit_transform(df1[col])
    
    df1 = df1.drop(columns=['ds1_f_37', 'ds1_f_40'])
    df2 = df2.drop(columns=['ds2_f_8', 'ds2_f_9', 'ds2_f_10', 'ds2_f_6', 
                           'ds2_f_24', 'ds2_f_11', 'ds2_f_12', 'ds2_f_13', 'ds2_f_23'])
    
    categorical_cols = ['ds2_f_1', 'ds2_f_4', 'ds2_f_5', 'ds2_f_7', 'ds2_f_8', 
                       'ds2_f_27', 'ds2_f_29', 'ds2_f_9', 'ds2_f_10', 'ds2_f_32']
    
    for col in categorical_cols:
        if col in df2.columns:
            le = LabelEncoder()
            df2[col] = le.fit_transform(df2[col].astype(str))
            freq = df2[col].value_counts()
            df2[f'{col}_freq'] = df2[col].map(freq)
    
    final_df = df1.merge(df2, on='mid', how='inner')
    final_df = final_df.merge(df3, on='mid', how='inner')
    
    return final_df

def load_data(in_folder: str):
    """
    Load and prepare the data from input folder
    """
    df_train = pd.read_parquet(os.path.join(in_folder, 'train_ids.parquet'))
    df_test = pd.read_parquet(os.path.join(in_folder, 'test_ids.parquet'))
    df1 = pd.read_parquet(os.path.join(in_folder, 'data_source_1.parquet'))
    df2 = pd.read_parquet(os.path.join(in_folder, 'data_source_2.parquet'))
    df3 = pd.read_parquet(os.path.join(in_folder, 'data_source_3.parquet'))
    
    n_rows = min(len(df1), len(df2), len(df3), len(df_train))
    df1 = df1.iloc[:n_rows]
    df2 = df2.iloc[:n_rows]
    df3 = df3.iloc[:n_rows]
    df_train = df_train.iloc[:n_rows]
    
    X = preprocess_data(df1, df2, df3)
    y = df_train['target_label']
    
    X_test = X[X['mid'].isin(df_test['mid'])]
    y_test = df_test.set_index('mid').loc[X_test['mid']]['target_label']
    
    X_train = X[~X['mid'].isin(df_test['mid'])]
    y_train = y[~X['mid'].isin(df_test['mid'])]
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, X_test, y_test):
    """
    Train the LightGBM model with improved class balancing and parameters
    """

    n_samples = len(y_train)
    n_positive = sum(y_train)
    n_negative = n_samples - n_positive
    scale_pos_weight = n_negative / n_positive

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': 8, 
        'min_data_in_leaf': 10, 
        'scale_pos_weight': scale_pos_weight * 2, 
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
    )
    
    return model

def train(in_folder: str, out_folder: str) -> None:
    """
    Main training pipeline
    """
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(in_folder)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X_train, y_train)
    
    print("Training model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print(f"""{'-' * 50}\nSaving Model ...""")
    save_model(model, out_folder)
    
    print("Training completed successfully!")

if __name__ == '__main__':
    fire.Fire(train)