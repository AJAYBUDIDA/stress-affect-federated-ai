"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Feature Extraction Module (Optimized Vectorized Operations)
"""

import math
import numpy as np

def calculate_stats(signal_slice):
    """
    Safely calculates statistical features (mean, std, min, max) for a 1D numerical slice.
    Uses fast numpy vectorized math. Handles empty arrays and NaNs safely.
    """
    if signal_slice is None or len(signal_slice) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    arr = np.asarray(signal_slice, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr)) if arr.size > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    
    return mean_val, std_val, min_val, max_val


def calculate_3d_acc_stats(acc_x_slice, acc_y_slice, acc_z_slice):
    """
    Calculates statistical features for 3-axis accelerometer data and magnitude vector.
    """
    x_mean, x_std, x_min, x_max = calculate_stats(acc_x_slice)
    y_mean, y_std, y_min, y_max = calculate_stats(acc_y_slice)
    z_mean, z_std, z_min, z_max = calculate_stats(acc_z_slice)
    
    if acc_x_slice is not None and acc_y_slice is not None and acc_z_slice is not None and len(acc_x_slice) > 0:
        x = np.asarray(acc_x_slice, dtype=np.float64)
        y = np.asarray(acc_y_slice, dtype=np.float64)
        z = np.asarray(acc_z_slice, dtype=np.float64)
        min_len = min(len(x), len(y), len(z))
        if min_len > 0:
            mag = np.sqrt(x[:min_len]**2 + y[:min_len]**2 + z[:min_len]**2)
            mag_mean, mag_std, mag_min, mag_max = calculate_stats(mag)
        else:
            mag_mean, mag_std, mag_min, mag_max = 0.0, 0.0, 0.0, 0.0
    else:
        mag_mean, mag_std, mag_min, mag_max = 0.0, 0.0, 0.0, 0.0
        
    return {
        'x_mean': x_mean, 'x_std': x_std, 'x_min': x_min, 'x_max': x_max,
        'y_mean': y_mean, 'y_std': y_std, 'y_min': y_min, 'y_max': y_max,
        'z_mean': z_mean, 'z_std': z_std, 'z_min': z_min, 'z_max': z_max,
        'mag_mean': mag_mean, 'mag_std': mag_std, 'mag_min': mag_min, 'mag_max': mag_max
    }


def extract_features_from_window(chest_window, wrist_window):
    """
    Extracts all required statistical features from chest and wrist signal windows.
    """
    features = {}
    
    # 1. Chest ECG
    f_mean, f_std, f_min, f_max = calculate_stats(chest_window.get('ECG'))
    features['c_ecg_mean'] = f_mean
    features['c_ecg_std'] = f_std
    features['c_ecg_min'] = f_min
    features['c_ecg_max'] = f_max
    
    # 2. Chest EDA
    f_mean, f_std, f_min, f_max = calculate_stats(chest_window.get('EDA'))
    features['c_eda_mean'] = f_mean
    features['c_eda_std'] = f_std
    features['c_eda_min'] = f_min
    features['c_eda_max'] = f_max
    
    # 3. Chest Temperature
    f_mean, f_std, f_min, f_max = calculate_stats(chest_window.get('Temp'))
    features['c_temp_mean'] = f_mean
    features['c_temp_std'] = f_std
    features['c_temp_min'] = f_min
    features['c_temp_max'] = f_max
    
    # 4. Chest Respiration
    f_mean, f_std, f_min, f_max = calculate_stats(chest_window.get('Resp'))
    features['c_resp_mean'] = f_mean
    features['c_resp_std'] = f_std
    features['c_resp_min'] = f_min
    features['c_resp_max'] = f_max
    
    # 5. Chest EMG
    f_mean, f_std, f_min, f_max = calculate_stats(chest_window.get('EMG'))
    features['c_emg_mean'] = f_mean
    features['c_emg_std'] = f_std
    features['c_emg_min'] = f_min
    features['c_emg_max'] = f_max
    
    # 6. Chest ACC (3-axis)
    c_acc_stats = calculate_3d_acc_stats(chest_window.get('ACC_X'), chest_window.get('ACC_Y'), chest_window.get('ACC_Z'))
    for k, v in c_acc_stats.items():
        features[f'c_acc_{k}'] = v
        
    # 7. Wrist BVP
    f_mean, f_std, f_min, f_max = calculate_stats(wrist_window.get('BVP'))
    features['w_bvp_mean'] = f_mean
    features['w_bvp_std'] = f_std
    features['w_bvp_min'] = f_min
    features['w_bvp_max'] = f_max
    
    # 8. Wrist EDA
    f_mean, f_std, f_min, f_max = calculate_stats(wrist_window.get('EDA'))
    features['w_eda_mean'] = f_mean
    features['w_eda_std'] = f_std
    features['w_eda_min'] = f_min
    features['w_eda_max'] = f_max
    
    # 9. Wrist TEMP
    f_mean, f_std, f_min, f_max = calculate_stats(wrist_window.get('TEMP'))
    features['w_temp_mean'] = f_mean
    features['w_temp_std'] = f_std
    features['w_temp_min'] = f_min
    features['w_temp_max'] = f_max
    
    # 10. Wrist HR
    f_mean, f_std, f_min, f_max = calculate_stats(wrist_window.get('HR'))
    features['w_hr_mean'] = f_mean
    features['w_hr_std'] = f_std
    features['w_hr_min'] = f_min
    features['w_hr_max'] = f_max
    
    # 11. Wrist ACC (3-axis)
    w_acc_stats = calculate_3d_acc_stats(wrist_window.get('ACC_X'), wrist_window.get('ACC_Y'), wrist_window.get('ACC_Z'))
    for k, v in w_acc_stats.items():
        features[f'w_acc_{k}'] = v
        
    return features
