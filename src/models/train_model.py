import pandas as pd
import numpy as np
import os
import joblib
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.paths import FEATURES_DATA_DIR, MODELS_DIR

def train_and_evaluate(features_csv_path, model_save_path):
    print(f"1. Dang tai du lieu tu {features_csv_path}...")
    df = pd.read_csv(features_csv_path)
    
    print(f"   Tong so mau: {len(df)}")
    
    # Kiem tra missing values
    if df.isnull().values.any():
        print("   Canh bao: Du lieu co chua NaN, se tien hanh drop.")
        df = df.dropna()
        
    print(f"   So luong mau sau khi clean: {len(df)}")
    
    # Phan tach Features (X) va Label (y)
    # y=0 la binh thuong, y=1 la Rung tam nhi (AFib)
    X = df.drop(columns=['status'])
    y = df['status']
    
    print("   Phan bo nhan (Class distribution):")
    print(y.value_counts())
    
    # Chia tap Train/Test (80% Train, 20% Test)
    print("\n2. Chia tap du lieu Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Khoi tao mo hinh Random Forest
    # Su dung class_weight='balanced' de xu ly mat can bang du lieu (imbalanced data)
    print("\n3. Huan luyen mo hinh Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    
    # Train
    rf_model.fit(X_train, y_train)
    
    # Du doan tren tap test
    print("\n4. Danh gia mo hinh...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   Accuracy: {acc:.4f}")
    
    print("\n=== Bao cao Phan loai (Classification Report) ===")
    print(classification_report(y_test, y_pred, target_names=['Binh Thuong (0)', 'AFib (1)']))
    
    # In ra cac feature quan trong nhat
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    print("\n=== Muc do quan trong cua cac dac trung (Top 5) ===")
    print(feat_imp.head(5))
    
    # Luu mo hinh
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(rf_model, model_save_path)
    print(f"\n5. Hoan thanh! Da luu mo hinh tai: {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Random Forest Model")
    parser.add_argument('--input', type=str, default=str(FEATURES_DATA_DIR / 'train_features.csv'), help="Path to features CSV file")
    parser.add_argument('--output', type=str, default=str(MODELS_DIR / 'afib_rf_model.pkl'), help="Path to save trained model")
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        train_and_evaluate(args.input, args.output)
    else:
        print(f"Loi: Khong tim thay file {args.input}. Vui long chay make_dataset.py truoc.")
