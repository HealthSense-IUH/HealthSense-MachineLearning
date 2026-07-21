import pandas as pd
import numpy as np
import os
import sys
import argparse
import joblib
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.paths import RAW_DATA_DIR, MODELS_DIR
from src.features.build_features import extract_features_from_window

def butter_bandpass_filter(data, lowcut=0.5, highcut=4.0, fs=125.0, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def process_raw_ppg_and_predict(raw_csv_path, model_path):
    print(f"=== HE THONG CANH BAO RUNG TAM NHI (AFib) ===")
    print(f"1. Doc du lieu thiet bi: {raw_csv_path}")
    
    df_raw = pd.read_csv(raw_csv_path)
    
    # Kiem tra thoi luong (Bo qua 3 giay dau, lay 60 giay tiep theo)
    start_time_ms = df_raw['Time(ms)'].iloc[0]
    df_raw_1min = df_raw[(df_raw['Time(ms)'] > start_time_ms + 3000) & (df_raw['Time(ms)'] <= start_time_ms + 63000)].copy()
    df_raw_1min['Time(s)'] = (df_raw_1min['Time(ms)'] - start_time_ms - 3000) / 1000.0
    
    # Chuan hoa thoi gian, xoa trung lap neu co
    df_raw_1min = df_raw_1min.drop_duplicates(subset=['Time(s)'])
    
    print(f"2. Tien xu ly tin hieu (Preprocessing)...")
    # Buoc 1: Noi suy 125Hz
    max_time_s = df_raw_1min['Time(s)'].max()
    new_time_s = np.arange(0, max_time_s, 0.008) # 125Hz
    
    interpolator = interp1d(df_raw_1min['Time(s)'], df_raw_1min['IR'], kind='cubic', fill_value="extrapolate")
    interpolated_ir = interpolator(new_time_s)
    
    # Buoc 2: Dao chieu (Inversion)
    inverted_ir = -interpolated_ir
    
    # Buoc 3: Loc nhieu (Bandpass Filter)
    filtered_ir = butter_bandpass_filter(inverted_ir)
    
    # Buoc 4: Chuan hoa (Min-Max) [0, 1] cho khop voi tham so tim dinh Peak
    f_min = np.min(filtered_ir)
    f_max = np.max(filtered_ir)
    normalized_ir = (filtered_ir - f_min) / (f_max - f_min)
    
    print(f"3. Trich xuat 13 Dac trung (Feature Extraction)...")
    features = extract_features_from_window(new_time_s, normalized_ir, fs=125.0)
    
    # Loai bo status neu co
    if 'status' in features:
        del features['status']
        
    # Kiem tra xem co the trich xuat features khong (co the do nhieu ko tim duoc tim)
    if pd.isna(features['HR_mean']):
        print("-> [LOI] Khong the trich xuat nhip tim. Vui long deo chat cam bien va khong cu dong tay!")
        return
        
    print(f"   - Nhip tim trung binh (HR): {features['HR_mean']:.0f} BPM")
    print(f"   - Bien thien nhip tim (SDNN): {features['SDNN']:.2f} ms")
    print(f"   - Can bang than kinh (LF/HF): {features['LF_HF_Ratio']:.2f}")
    
    print(f"4. Du doan bang AI Model...")
    rf_model = joblib.load(model_path)
    
    # Chuyen dictionary thanh DataFrame co dung thu tu cot cua training
    features_df = pd.DataFrame([features])
    
    prediction = rf_model.predict(features_df)[0]
    
    print("\n================ KET QUA ================")
    if prediction == 1:
        print("[CANH BAO] PHAT HIEN DAU HIEU RUNG TAM NHI (AFib)!")
        print("   Nhip tim cua ban co su bien thien bat thuong.")
    else:
        print("[BINH THUONG] Nhip tim on dinh. Khong phat hien Rung tam nhi.")
    print("=========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict AFib from raw PPG data")
    
    # Mặc định lấy 1 file mẫu từ data/raw để test
    default_raw = str(RAW_DATA_DIR / 'do_20260713_215444_session1.csv')
    default_model = str(MODELS_DIR / 'afib_rf_model.pkl')
    
    parser.add_argument('--input', type=str, default=default_raw, help="Path to raw CSV file")
    parser.add_argument('--model', type=str, default=default_model, help="Path to trained model")
    args = parser.parse_args()
    
    if os.path.exists(args.input) and os.path.exists(args.model):
        process_raw_ppg_and_predict(args.input, args.model)
    else:
        print(f"Khong tim thay file data ({args.input}) hoac model ({args.model}).")
