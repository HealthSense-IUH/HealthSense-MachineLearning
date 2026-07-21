import numpy as np
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d

def get_rr_intervals(time_array, ppg_array, fs=125.0):
    """
    Tìm đỉnh (peaks) trên tín hiệu PPG và tính chuỗi R-R intervals.
    """
    # Tìm đỉnh (tâm thu). Các tham số được tinh chỉnh cho tín hiệu PPG 125Hz
    # Khoảng cách giữa 2 nhịp tim tối thiểu ~ 250ms (tương đương max 240 BPM)
    min_distance = int(0.25 * fs) 
    
    peaks, _ = find_peaks(ppg_array, distance=min_distance, prominence=0.05)
    
    if len(peaks) < 2:
        return np.array([])
        
    peak_times = time_array[peaks]
    # R-R intervals in milliseconds
    rr_intervals = np.diff(peak_times) * 1000.0
    return rr_intervals

def calculate_time_domain(rr_intervals):
    """
    Trích xuất 6 đặc trưng Time-Domain
    """
    if len(rr_intervals) < 2:
        return {k: np.nan for k in ['Mean_NN', 'SDNN', 'RMSSD', 'pNN50', 'NN50', 'CV']}
    
    mean_nn = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals, ddof=1)
    
    diff_nn = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))
    
    nn50 = np.sum(np.abs(diff_nn) > 50)
    pnn50 = (nn50 / len(diff_nn)) * 100 if len(diff_nn) > 0 else 0
    
    cv = (sdnn / mean_nn) * 100 if mean_nn > 0 else np.nan
    
    return {
        'Mean_NN': mean_nn,
        'SDNN': sdnn,
        'RMSSD': rmssd,
        'pNN50': pnn50,
        'NN50': nn50,
        'CV': cv
    }

def calculate_frequency_domain(rr_intervals, fs_interp=4.0):
    """
    Trích xuất 6 đặc trưng Frequency-Domain
    """
    nan_result = {k: np.nan for k in ['LF', 'HF', 'LF_HF_Ratio', 'LF_norm', 'HF_norm', 'Total_Power']}
    if len(rr_intervals) < 10:
        return nan_result
    
    # Tạo trục thời gian tích lũy (giây)
    t = np.cumsum(rr_intervals) / 1000.0
    t -= t[0]
    
    # R-R intervals cần ở đơn vị giây cho phân tích tần số chuẩn
    rr_sec = rr_intervals / 1000.0
    
    # Cần nội suy để tạo tín hiệu đều (evenly sampled)
    try:
        f_interp = interp1d(t, rr_sec, kind='cubic')
        t_interp = np.arange(t[0], t[-1], 1.0 / fs_interp)
        rr_interp = f_interp(t_interp)
    except Exception:
        return nan_result
        
    # Welch's periodogram (PSD)
    nperseg = min(256, len(rr_interp))
    if nperseg < 2:
        return nan_result
        
    f, pxx = welch(rr_interp, fs=fs_interp, nperseg=nperseg)
    
    # LF (0.04 - 0.15 Hz) và HF (0.15 - 0.4 Hz)
    lf_band = (f >= 0.04) & (f <= 0.15)
    hf_band = (f >= 0.15) & (f <= 0.4)
    
    # Tính tích phân (diện tích dưới đường cong)
    df = f[1] - f[0]
    lf = np.sum(pxx[lf_band]) * df
    hf = np.sum(pxx[hf_band]) * df
    
    total_power = lf + hf
    lf_hf_ratio = lf / hf if hf > 0 else np.nan
    
    lf_norm = (lf / total_power) * 100 if total_power > 0 else np.nan
    hf_norm = (hf / total_power) * 100 if total_power > 0 else np.nan
    
    return {
        'LF': lf,
        'HF': hf,
        'LF_HF_Ratio': lf_hf_ratio,
        'LF_norm': lf_norm,
        'HF_norm': hf_norm,
        'Total_Power': total_power
    }

def extract_features_from_window(time_array, ppg_array, fs=125.0):
    """
    Trích xuất toàn bộ 13 đặc trưng từ một cửa sổ thời gian (VD: 60s)
    """
    rr_intervals = get_rr_intervals(time_array, ppg_array, fs)
    
    # Khởi tạo kết quả mặc định (nếu lỗi hoặc nhiễu không tìm thấy tim)
    features = {
        'HR_mean': np.nan,
        'Mean_NN': np.nan, 'SDNN': np.nan, 'RMSSD': np.nan, 
        'pNN50': np.nan, 'NN50': np.nan, 'CV': np.nan,
        'LF': np.nan, 'HF': np.nan, 'LF_HF_Ratio': np.nan,
        'LF_norm': np.nan, 'HF_norm': np.nan, 'Total_Power': np.nan
    }
    
    if len(rr_intervals) > 0:
        # Heart Rate
        features['HR_mean'] = 60000.0 / np.mean(rr_intervals)
        
        # Time Domain
        td_features = calculate_time_domain(rr_intervals)
        features.update(td_features)
        
        # Frequency Domain
        fd_features = calculate_frequency_domain(rr_intervals)
        features.update(fd_features)
        
    return features
