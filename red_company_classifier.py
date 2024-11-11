import fire
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def load_data(in_folder: str):
    """
    Load and combine data from multiple sources
    """
    train_ids = pd.read_parquet(os.path.join(in_folder, 'train_ids.parquet'))
    test_ids = pd.read_parquet(os.path.join(in_folder, 'test_ids.parquet'))
    
    ds1 = pd.read_parquet(os.path.join(in_folder, 'data_source_1.parquet'))
    ds2 = pd.read_parquet(os.path.join(in_folder, 'data_source_2.parquet'))
    ds3 = pd.read_parquet(os.path.join(in_folder, 'data_source_3.parquet'))
    
    data = ds1.merge(ds2, on='mid', how='left')
    data = data.merge(ds3, on='mid', how='left')
    
    train_data = data[data['mid'].isin(train_ids['mid'])]
    train_data = train_data.merge(train_ids[['mid', 'target_label']], on='mid', how='left')
    
    test_data = data[data['mid'].isin(test_ids['mid'])]
    test_data = test_data.merge(test_ids[['mid', 'target_label']], on='mid', how='left')
    
    return train_data, test_data

def preprocess_data(data, train_columns=None):
    """
    Clean and prepare data for modeling
    Parameters:
        data: DataFrame to process
        train_columns: List of columns from training data (used for test data to ensure consistency)
    """
    X = data.drop(['mid', 'target_label'], axis=1)
    y = data['target_label']
    
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    
    all_dummies = []
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])
        dummies = pd.get_dummies(X[col], prefix=col)
        all_dummies.append(dummies)
    
    X = pd.concat([X[numeric_cols]] + all_dummies, axis=1)
    
    if train_columns is not None:
        new_X = pd.DataFrame(0, index=X.index, columns=train_columns)
        
        common_cols = X.columns.intersection(train_columns)
        new_X[common_cols] = X[common_cols]
        
        X = new_X
    
    return X, y

def split_data(data):
    """
    Split data into train and validation sets
    """
    X, y = preprocess_data(data)
    global train_columns
    train_columns = X.columns
    return train_test_split(X, y, test_size=0.3, random_state=60)

def evaluate_model(model, test_data):
    """
    Evaluate model performance
    """
    X_test, y_test = preprocess_data(test_data, train_columns)
    y_pred = model.predict(X_test)
    print("\nModel Evaluation Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, out_folder: str):
    """
    Save the trained model
    """
    os.makedirs(out_folder, exist_ok=True)
    joblib.dump(model, os.path.join(out_folder, 'model.joblib'))

def train(in_folder: str, out_folder: str) -> None:
    """
    Main training pipeline
    """
    print("Loading data...")
    train_data, test_data = load_data(in_folder)
    
    print("Splitting data...")
    X_train, X_val, y_train, y_val = split_data(train_data)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, test_data)
    
    print("Saving model...")
    save_model(model, out_folder)
    print(f"Model saved to {out_folder}")

if __name__ == '__main__':
    fire.Fire(train)