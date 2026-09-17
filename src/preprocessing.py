"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Preprocessing & Dataset Windowing Pipeline (Fast Vectorized NumPy Implementation)
"""

import os
import sys
import csv
import math
import pickle
import gc
import numpy as np

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import extract_features_from_window

# Target Label Mapping: 1=Neutral, 2=Stress, 3=Amusement
LABEL_MAP = {
    1: 'Neutral',
    2: 'Stress',
    3: 'Amusement'
}

def load_e4_hr_data(subject_dir):
    """Loads HR values from S<ID>_E4_Data/HR.csv if available."""
    hr_file = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}_E4_Data", "HR.csv")
    if os.path.exists(hr_file):
        try:
            arr = np.genfromtxt(hr_file, skip_header=2, delimiter=',')
            return arr if arr.ndim == 1 else arr.flatten()
        except Exception as e:
            print(f"  Warning: Could not read HR.csv for {os.path.basename(subject_dir)}: {e}")
    return None


def extract_slice(arr, start_idx, end_idx, is_3d=False):
    """Safely extracts NumPy slice or 3-axis accelerometer tuple."""
    if arr is None or not hasattr(arr, 'shape'):
        return (None, None, None) if is_3d else None
        
    if is_3d:
        if len(arr.shape) == 2 and arr.shape[1] == 3:
            sub = arr[start_idx:end_idx]
            return sub[:, 0], sub[:, 1], sub[:, 2]
        return None, None, None
    else:
        sub = arr[start_idx:end_idx]
        if len(sub.shape) > 1:
            sub = sub.flatten()
        return sub


def process_single_subject(subject_id, wesad_raw_dir, window_sec=5.0):
    """
    Processes a single WESAD subject file memory-efficiently using vectorized NumPy operations.
    Yields dict rows of extracted features for valid 5-second windows.
    """
    subject_dir = os.path.join(wesad_raw_dir, subject_id)
    pkl_path = os.path.join(subject_dir, f"{subject_id}.pkl")
    
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} does not exist.")
        return
        
    print(f"Processing Subject: {subject_id}...")
    with open(pkl_path, 'rb') as f:
        subj_data = pickle.load(f, encoding='latin1')
        
    chest_signals = subj_data.get('signal', {}).get('chest', {})
    wrist_signals = subj_data.get('signal', {}).get('wrist', {})
    label_arr = subj_data.get('label')
    
    # Load E4 HR vector
    wrist_hr_arr = load_e4_hr_data(subject_dir)
    
    c_ecg = chest_signals.get('ECG')
    total_chest_samples = c_ecg.shape[0] if c_ecg is not None else 0
    
    samples_per_window_700 = int(window_sec * 700) # 3500 samples
    total_windows = total_chest_samples // samples_per_window_700
    
    valid_window_count = 0
    
    for win_idx in range(total_windows):
        c_start = win_idx * samples_per_window_700
        c_end = c_start + samples_per_window_700
        
        # 1. Ground Truth Label Validation
        lbl_slice = label_arr[c_start:c_end]
        if lbl_slice is None or len(lbl_slice) == 0:
            continue
            
        # Fast label majority count using np.bincount
        int_lbls = lbl_slice.astype(np.int32).flatten()
        counts = np.bincount(int_lbls)
        majority_label = int(np.argmax(counts))
        majority_ratio = counts[majority_label] / len(int_lbls)
        
        # Keep window ONLY if majority label is Neutral (1), Stress (2), or Amusement (3) with >=85% purity
        if majority_label not in LABEL_MAP or majority_ratio < 0.85:
            continue
            
        label_name = LABEL_MAP[majority_label]
        
        # Timestamps in seconds
        t_start_sec = c_start / 700.0
        t_end_sec = c_end / 700.0
        
        # 2. Chest Signal Windows (700 Hz)
        c_win = {
            'ECG': extract_slice(chest_signals.get('ECG'), c_start, c_end),
            'EDA': extract_slice(chest_signals.get('EDA'), c_start, c_end),
            'Temp': extract_slice(chest_signals.get('Temp'), c_start, c_end),
            'Resp': extract_slice(chest_signals.get('Resp'), c_start, c_end),
            'EMG': extract_slice(chest_signals.get('EMG'), c_start, c_end)
        }
        c_x, c_y, c_z = extract_slice(chest_signals.get('ACC'), c_start, c_end, is_3d=True)
        c_win['ACC_X'], c_win['ACC_Y'], c_win['ACC_Z'] = c_x, c_y, c_z
        
        # 3. Wrist Signal Windows (Native sampling rates)
        # ACC (32 Hz)
        w_acc_start, w_acc_end = int(t_start_sec * 32), int(t_end_sec * 32)
        w_x, w_y, w_z = extract_slice(wrist_signals.get('ACC'), w_acc_start, w_acc_end, is_3d=True)
        
        # BVP (64 Hz)
        w_bvp_start, w_bvp_end = int(t_start_sec * 64), int(t_end_sec * 64)
        w_bvp = extract_slice(wrist_signals.get('BVP'), w_bvp_start, w_bvp_end)
        
        # EDA (4 Hz)
        w_eda_start, w_eda_end = int(t_start_sec * 4), int(t_end_sec * 4)
        w_eda = extract_slice(wrist_signals.get('EDA'), w_eda_start, w_eda_end)
        
        # TEMP (4 Hz)
        w_temp_start, w_temp_end = int(t_start_sec * 4), int(t_end_sec * 4)
        w_temp = extract_slice(wrist_signals.get('TEMP'), w_temp_start, w_temp_end)
        
        # HR (1 Hz)
        w_hr_start, w_hr_end = int(t_start_sec * 1), int(t_end_sec * 1)
        w_hr = extract_slice(wrist_hr_arr, w_hr_start, w_hr_end)
        
        w_win = {
            'BVP': w_bvp,
            'EDA': w_eda,
            'TEMP': w_temp,
            'HR': w_hr,
            'ACC_X': w_x, 'ACC_Y': w_y, 'ACC_Z': w_z
        }
        
        # 4. Feature Extraction
        feature_dict = extract_features_from_window(c_win, w_win)
        
        # 5. Metadata Row
        row = {
            'subject_id': subject_id,
            'label': majority_label,
            'label_name': label_name,
            'window_id': f"{subject_id}_w{win_idx:04d}"
        }
        row.update(feature_dict)
        
        valid_window_count += 1
        yield row
        
    print(f"  Finished {subject_id}: {valid_window_count} valid windows extracted.")
    
    # Garbage collection
    del subj_data, chest_signals, wrist_signals, label_arr
    gc.collect()


def run_preprocessing_pipeline(wesad_raw_dir, output_csv):
    """
    Runs full preprocessing pipeline across all 15 subjects.
    Saves feature dataset to output_csv.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    subject_ids = sorted([
        d for d in os.listdir(wesad_raw_dir)
        if os.path.isdir(os.path.join(wesad_raw_dir, d)) and d.startswith('S') and d[1:].isdigit()
    ], key=lambda x: int(x[1:]))
    
    print(f"Found {len(subject_ids)} subjects to process: {subject_ids}")
    
    total_rows = 0
    header_written = False
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = None
        for subj_id in subject_ids:
            for row in process_single_subject(subj_id, wesad_raw_dir):
                if not header_written:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True
                writer.writerow(row)
                total_rows += 1
                
    print("\nProcessing complete!")
    print(f"Extracted {total_rows} feature windows saved to {output_csv}")
    
    # Generate sample validation file (first 50 rows)
    sample_csv = output_csv.replace(".csv", "_sample.csv")
    with open(output_csv, 'r', encoding='utf-8') as f_in, open(sample_csv, 'w', newline='', encoding='utf-8') as f_sample:
        for i in range(51):
            line = f_in.readline()
            if not line:
                break
            f_sample.write(line)
    print(f"Saved sample validation file to {sample_csv}")


if __name__ == "__main__":
    wesad_dir = r"d:\stress-affect-federated-ai\data\raw\WESAD"
    output_file = r"d:\stress-affect-federated-ai\data\processed\wesad_features.csv"
    run_preprocessing_pipeline(wesad_dir, output_file)
